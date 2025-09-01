use std::sync::Arc;

use anyhow::Result;
use egui::viewport::ViewportId;
use egui::Context as EguiContext;
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use glam::{Mat4, Quat, Vec3};
use web_time::Instant;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::window::Window;

use include_dir::{include_dir, Dir};
use serde::Deserialize;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Convert OpenGL clip space depth range (-1..1) to WebGPU/Vulkan range (0..1)
// See wgpu examples for the same transform.
const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
]);
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct RotationDeg {
    #[serde(default)]
    yaw: f32,
    #[serde(default)]
    pitch: f32,
    #[serde(default)]
    roll: f32,
}

#[derive(Debug, Deserialize)]
struct SceneDesc {
    name: String,
    geometry: String, // "cube" | "quad"
    shader: String,   // WGSL file name in the scene folder
    #[serde(default)]
    rotation_speed: Option<f32>,
    #[serde(default)]
    rotation_deg: Option<RotationDeg>, // initial orientation (degrees)
}

struct EngineScene {
    name: String,
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    num_indices: u32,
    rotation_speed: f32,
    rotation_deg: Option<RotationDeg>,
}

// Embed all scene folders from engine/scenes/
static SCENES_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/scenes");

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
}

struct Offscreen {
    size: PhysicalSize<u32>,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    texture_id: Option<egui::TextureId>,
}

impl Offscreen {
    fn new(device: &wgpu::Device, size: PhysicalSize<u32>) -> Self {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_color"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());

        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_depth"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            size,
            color,
            color_view,
            depth,
            depth_view,
            texture_id: None,
        }
    }
}

pub struct EngineApp {
    pub window: Arc<Window>,
    // gpu
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    surface_format: wgpu::TextureFormat,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    // egui
    egui_ctx: EguiContext,
    egui_state: EguiWinitState,
    egui_renderer: EguiRenderer,
    start_time: Instant,

    // scenes
    scenes: Vec<EngineScene>,
    current_scene: usize,

    // uniforms
    uniform_buf: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // offscreen viewport
    viewport: Offscreen,
    viewport_size_ui: [u32; 2],
    rotation_speed: f32,
}

impl EngineApp {
    pub async fn new(window: Window) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ =
                env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
                    .try_init();
        }
        #[cfg(target_arch = "wasm32")]
        {
            console_error_panic_hook::set_once();
            console_log::init_with_level(log::Level::Info).ok();
        }

        let window = Arc::new(window);

        // Instance + Surface
        let instance = wgpu::Instance::default();
        // Keep Arc<Window> in the struct to uphold safety for 'static surface
        let surface = unsafe {
            instance
                .create_surface_unsafe(
                    wgpu::SurfaceTargetUnsafe::from_window(window.as_ref())
                        .expect("window handle for surface"),
                )
                .expect("create surface")
        };
        // Robust adapter selection: try several options to avoid NotFound on some platforms/browsers
        let adapter = if let Ok(a) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            a
        } else if let Ok(a) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            a
        } else if let Ok(a) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            a
        } else if let Ok(a) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            a
        } else {
            return Err(anyhow::anyhow!(
                "No WebGPU adapter found. On web, ensure WebGPU is enabled in your browser (e.g., Chrome: chrome://flags/#enable-unsafe-webgpu) and that your GPU/browser supports WebGPU."
            ));
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps
                .present_modes
                .get(0)
                .copied()
                .unwrap_or(wgpu::PresentMode::AutoVsync),
            alpha_mode: caps
                .alpha_modes
                .get(0)
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // egui init
        let egui_ctx = EguiContext::default();
        let egui_state = EguiWinitState::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            Some(8192),
        );
        let egui_renderer = EguiRenderer::new(&device, surface_format, None, 1, false);

        // Offscreen viewport default
        let viewport_size_ui = [1920, 1080];
        let viewport = Offscreen::new(
            &device,
            PhysicalSize::new(viewport_size_ui[0], viewport_size_ui[1]),
        );

        // Uniforms
        let initial_mvp = Mat4::IDENTITY;
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                mvp: initial_mvp.to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let u_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("u_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("u_bg"),
            layout: &u_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&u_layout],
            push_constant_ranges: &[],
        });

        // Build scenes from embedded descriptors
        let scenes = Self::build_scenes(&device, &pipeline_layout);
        // Default to "Rainbow Cube" if it exists; otherwise first scene (or fallback)
        let mut current_scene = 0;
        if let Some((i, _)) = scenes
            .iter()
            .enumerate()
            .find(|(_, s)| s.name == "Rainbow Cube")
        {
            current_scene = i;
        }

        Ok(Self {
            window,
            instance,
            surface,
            surface_config,
            surface_format,
            adapter,
            device,
            queue,
            egui_ctx,
            egui_state,
            egui_renderer,
            start_time: Instant::now(),
            scenes,
            current_scene,
            uniform_buf,
            uniform_bind_group,
            viewport,
            viewport_size_ui,
            rotation_speed: 1.0,
        })
    }

    pub fn window(&self) -> Arc<Window> {
        self.window.clone()
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.egui_state
            .on_window_event(self.window.as_ref(), event)
            .consumed
    }

    fn ensure_viewport(&mut self, ui_size: [u32; 2]) {
        // Treat UI-entered size as physical pixels. Do NOT multiply by pixels_per_point here.
        let phys_w = ui_size[0].max(1);
        let phys_h = ui_size[1].max(1);

        if self.viewport.size.width != phys_w || self.viewport.size.height != phys_h {
            self.viewport = Offscreen::new(&self.device, PhysicalSize::new(phys_w, phys_h));
            // Re-register texture with egui renderer
            if let Some(id) = self.viewport.texture_id.take() {
                // Best-effort free; ignore if API differs
                #[allow(unused_must_use)]
                {
                    self.egui_renderer.free_texture(&id);
                }
            }
        }
        if self.viewport.texture_id.is_none() {
            let id = self.egui_renderer.register_native_texture(
                &self.device,
                &self.viewport.color_view,
                wgpu::FilterMode::Linear,
            );
            self.viewport.texture_id = Some(id);
        }
    }

    pub fn update(&mut self) {}

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        // Time and uniforms
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let angle = elapsed * self.rotation_speed;
        let aspect =
            self.viewport.size.width.max(1) as f32 / self.viewport.size.height.max(1) as f32;
        // Build GL-style projection then convert depth to WebGPU (0..1)
        let proj_gl = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 100.0);
        let camera_pos = Vec3::new(0., 0., 3.5);
        let view_m = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        // Base model orientation from scene config (degrees)
        let base_model = if let Some(rot) = self.scenes[self.current_scene].rotation_deg {
            let yaw = rot.yaw.to_radians();
            let pitch = rot.pitch.to_radians();
            let roll = rot.roll.to_radians();
            Mat4::from_rotation_y(yaw) * Mat4::from_rotation_x(pitch) * Mat4::from_rotation_z(roll)
        } else {
            Mat4::IDENTITY
        };
        // Animated component (respects rotation_speed; zero means static)
        let anim_model =
            Mat4::from_rotation_y(angle) * Mat4::from_quat(Quat::from_rotation_x(angle * 0.7));
        let model = base_model * anim_model;
        let mvp = OPENGL_TO_WGPU_MATRIX * proj_gl * view_m * model;
        self.queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&Uniforms {
                mvp: mvp.to_cols_array_2d(),
            }),
        );

        // Ensure viewport exists for displaying previous frame
        self.ensure_viewport(self.viewport_size_ui);

        // Render cube into offscreen
        {
            let scene = &self.scenes[self.current_scene];
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cube_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.viewport.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.viewport.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&scene.pipeline);
            rpass.set_bind_group(0, &self.uniform_bind_group, &[]);
            rpass.set_vertex_buffer(0, scene.vbuf.slice(..));
            rpass.set_index_buffer(scene.ibuf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..scene.num_indices, 0, 0..1);
        }

        // Run egui
        let raw_input = self.egui_state.take_egui_input(self.window.as_ref());

        // Copy UI-controlled state to locals to avoid borrowing self in closure
        let mut ui_viewport_size = self.viewport_size_ui;
        let mut ui_rotation_speed = self.rotation_speed;
        let mut ui_scene_index = self.current_scene;
        let display_tex = self.viewport.texture_id;

        // Capture current DPI scaling for sizing the displayed image in points
        let ppp = self.egui_ctx.pixels_per_point();
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::TopBottomPanel::top("top").show(ctx, |ui| {
                ui.heading("Demo Engine");
            });
            egui::SidePanel::left("left")
                .resizable(true)
                .show(ctx, |ui| {
                    egui::ComboBox::from_label("Scene")
                        .selected_text(self.scenes[ui_scene_index].name.clone())
                        .show_ui(ui, |ui| {
                            for (i, s) in self.scenes.iter().enumerate() {
                                let selected = i == ui_scene_index;
                                if ui.selectable_label(selected, &s.name).clicked() {
                                    ui_scene_index = i;
                                }
                            }
                        });
                    ui.label("Viewport size (pixels)");
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut ui_viewport_size[0]).range(1..=4096));
                        ui.add(egui::DragValue::new(&mut ui_viewport_size[1]).range(1..=4096));
                    });
                    ui.label("Rotation speed");
                    ui.add(egui::Slider::new(&mut ui_rotation_speed, 0.0..=4.0));
                });
            egui::CentralPanel::default().show(ctx, |ui| {
                if let Some(id) = display_tex {
                    // Divide by pixels_per_point so that on-screen size in points maps to the desired physical pixels
                    let size = egui::Vec2::new(
                        ui_viewport_size[0] as f32 / ppp,
                        ui_viewport_size[1] as f32 / ppp,
                    );
                    ui.image(egui::load::SizedTexture { id, size });
                } else {
                    ui.label("No viewport");
                }
            });
        });

        // Apply any UI changes and ensure viewport for next frame
        self.viewport_size_ui = ui_viewport_size;
        if ui_scene_index != self.current_scene {
            self.current_scene = ui_scene_index;
            // Adopt the scene's suggested rotation speed if provided
            self.rotation_speed = self.scenes[self.current_scene].rotation_speed;
        } else {
            self.rotation_speed = ui_rotation_speed;
        }
        self.ensure_viewport(self.viewport_size_ui);
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, self.egui_ctx.pixels_per_point());

        // Upload egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: self.egui_ctx.pixels_per_point(),
        };
        // Prepare egui GPU buffers using a separate encoder to keep the main encoder free
        let mut egui_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("egui_prepare"),
                });
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut egui_encoder,
            &paint_jobs,
            &screen_desc,
        );
        self.queue.submit(std::iter::once(egui_encoder.finish()));

        // Render egui onto surface
        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.025,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut rpass = rpass.forget_lifetime();
            self.egui_renderer
                .render(&mut rpass, &paint_jobs, &screen_desc);
        }

        // Cleanup egui textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn build_scenes(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
    ) -> Vec<EngineScene> {
        let mut out = Vec::new();
        let scene_dirs: Vec<_> = SCENES_DIR.dirs().collect();
        log::info!("Scene scan: {} directories embedded", scene_dirs.len());
        for dir in scene_dirs {
            // Debug: list files discovered in this scene directory
            for f in dir.files() {
                log::info!("  file: {}", f.path().display());
            }
            // Parse descriptor
            let desc_file_opt = dir.get_file("scene.json").or_else(|| {
                dir.files()
                    .find(|f| f.path().file_name().and_then(|n| n.to_str()) == Some("scene.json"))
            });
            let Some(desc_file) = desc_file_opt else {
                log::warn!("Scene dir missing scene.json; skipping");
                continue;
            };
            let Some(json) = desc_file.contents_utf8() else {
                log::warn!("scene.json is not valid UTF-8; skipping");
                continue;
            };
            let desc: SceneDesc = match serde_json::from_str(json) {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("scene.json failed to parse: {}", e);
                    continue;
                }
            };
            log::info!("Scene descriptor parsed: {}", desc.name);

            // Load shader text
            let shader_file_opt = dir.get_file(desc.shader.as_str()).or_else(|| {
                dir.files().find(|f| {
                    f.path().file_name().and_then(|n| n.to_str()) == Some(desc.shader.as_str())
                })
            });
            let Some(shader_file) = shader_file_opt else {
                log::warn!(
                    "Shader '{}' not found in scene dir {}; skipping",
                    desc.shader,
                    desc.name
                );
                continue;
            };
            let Some(shader_src) = shader_file.contents_utf8() else {
                log::warn!("Shader '{}' is not valid UTF-8; skipping", desc.shader);
                continue;
            };
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("shader: {}", desc.name)),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

            // Geometry
            let (vbytes, ibytes, icount) = match desc.geometry.as_str() {
                "cube" => {
                    let vertices: &[Vertex] = &[
                        // front
                        Vertex {
                            position: [-0.5, -0.5, 0.5],
                            color: [1.0, 0.0, 0.0],
                        },
                        Vertex {
                            position: [0.5, -0.5, 0.5],
                            color: [0.0, 1.0, 0.0],
                        },
                        Vertex {
                            position: [0.5, 0.5, 0.5],
                            color: [0.0, 0.0, 1.0],
                        },
                        Vertex {
                            position: [-0.5, 0.5, 0.5],
                            color: [1.0, 1.0, 0.0],
                        },
                        // back
                        Vertex {
                            position: [-0.5, -0.5, -0.5],
                            color: [1.0, 0.0, 1.0],
                        },
                        Vertex {
                            position: [0.5, -0.5, -0.5],
                            color: [0.0, 1.0, 1.0],
                        },
                        Vertex {
                            position: [0.5, 0.5, -0.5],
                            color: [1.0, 0.5, 0.0],
                        },
                        Vertex {
                            position: [-0.5, 0.5, -0.5],
                            color: [0.5, 0.0, 0.5],
                        },
                    ];
                    let indices: &[u16] = &[
                        0, 1, 2, 2, 3, 0, // front (CCW)
                        5, 4, 7, 7, 6, 5, // back (CCW)
                        4, 0, 3, 3, 7, 4, // left (CCW)
                        1, 5, 6, 6, 2, 1, // right (CCW)
                        3, 2, 6, 6, 7, 3, // top (CCW)
                        4, 5, 1, 1, 0, 4, // bottom (CCW)
                    ];
                    (
                        bytemuck::cast_slice(vertices).to_vec(),
                        bytemuck::cast_slice(indices).to_vec(),
                        indices.len() as u32,
                    )
                }
                "quad" => {
                    // 1x1 square centered at origin facing +Z
                    let vertices: &[Vertex] = &[
                        Vertex {
                            position: [-0.5, -0.5, 0.0],
                            color: [1.0, 1.0, 1.0],
                        },
                        Vertex {
                            position: [0.5, -0.5, 0.0],
                            color: [1.0, 1.0, 1.0],
                        },
                        Vertex {
                            position: [0.5, 0.5, 0.0],
                            color: [1.0, 1.0, 1.0],
                        },
                        Vertex {
                            position: [-0.5, 0.5, 0.0],
                            color: [1.0, 1.0, 1.0],
                        },
                    ];
                    let indices: &[u16] = &[0, 1, 2, 2, 3, 0];
                    (
                        bytemuck::cast_slice(vertices).to_vec(),
                        bytemuck::cast_slice(indices).to_vec(),
                        indices.len() as u32,
                    )
                }
                other => {
                    log::warn!(
                        "Unrecognized geometry '{}'; skipping scene {}",
                        other,
                        desc.name
                    );
                    continue;
                }
            };

            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} vbuf", desc.name)),
                contents: &vbytes,
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} ibuf", desc.name)),
                contents: &ibytes,
                usage: wgpu::BufferUsages::INDEX,
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{} pipeline", desc.name)),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            out.push(EngineScene {
                name: desc.name,
                pipeline,
                vbuf,
                ibuf,
                num_indices: icount,
                rotation_speed: desc.rotation_speed.unwrap_or(1.0),
                rotation_deg: desc.rotation_deg,
            });
            log::info!(
                "Scene built: {}",
                out.last().map(|s| s.name.as_str()).unwrap_or("<unknown>")
            );
        }

        if out.is_empty() {
            // Fallback: build a single default cube scene using the built-in shader
            log::warn!("No scenes embedded; using built-in Default Cube");
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("default cube shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });
            let vertices: &[Vertex] = &[
                Vertex {
                    position: [-0.5, -0.5, 0.5],
                    color: [1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.5, 0.5],
                    color: [0.0, 1.0, 0.0],
                },
                Vertex {
                    position: [0.5, 0.5, 0.5],
                    color: [0.0, 0.0, 1.0],
                },
                Vertex {
                    position: [-0.5, 0.5, 0.5],
                    color: [1.0, 1.0, 0.0],
                },
                Vertex {
                    position: [-0.5, -0.5, -0.5],
                    color: [1.0, 0.0, 1.0],
                },
                Vertex {
                    position: [0.5, -0.5, -0.5],
                    color: [0.0, 1.0, 1.0],
                },
                Vertex {
                    position: [0.5, 0.5, -0.5],
                    color: [1.0, 0.5, 0.0],
                },
                Vertex {
                    position: [-0.5, 0.5, -0.5],
                    color: [0.5, 0.0, 0.5],
                },
            ];
            let indices: &[u16] = &[
                0, 1, 2, 2, 3, 0, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6,
                7, 3, 4, 5, 1, 1, 0, 4,
            ];
            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("default vbuf"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("default ibuf"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("default pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
            out.push(EngineScene {
                name: "Default Cube".into(),
                pipeline,
                vbuf,
                ibuf,
                num_indices: indices.len() as u32,
                rotation_speed: 1.0,
                rotation_deg: None,
            });
        }

        out
    }
}

// WGSL shader embedded alongside this file
// Note: kept simple and portable
#[allow(dead_code)]
const _: &str = include_str!("shader.wgsl");
