use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;
use wasm_bindgen_futures::spawn_local;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use std::cell::RefCell;

thread_local! {
    static ENGINE: RefCell<Option<engine::EngineApp>> = const { RefCell::new(None) };
}

struct App {
    window_id: Option<WindowId>,
}

impl Default for App {
    fn default() -> Self { Self { window_id: None } }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window_id.is_some() { return; }
        let attrs = Window::default_attributes().with_title("Demo Engine (Web)");
        let window = event_loop.create_window(attrs).expect("create window");
        // On web, ensure the canvas is attached to the DOM so WebGPU can create a context
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(canvas) = window.canvas() {
                if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                    if let Some(body) = doc.body() {
                        let _ = body.append_child(&canvas);
                    }
                }
            }
        }
        self.window_id = Some(window.id());

        // Initialize engine asynchronously on wasm
        spawn_local(async move {
            match engine::EngineApp::new(window).await {
                Ok(mut eng) => {
                    // trigger first redraw once ready
                    let win = eng.window();
                    ENGINE.with(|cell| { *cell.borrow_mut() = Some(eng); });
                    win.request_redraw();
                }
                Err(err) => {
                    web_sys::console::error_1(&format!("Engine init error: {err:?}").into());
                }
            }
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                match event {
                    WindowEvent::Resized(size) => engine.resize(size),
                    WindowEvent::ScaleFactorChanged { scale_factor: _, mut inner_size_writer } => {
                        // Keep current physical size so swapchain tracks DPI
                        let size = engine.window().inner_size();
                        let _ = inner_size_writer.request_inner_size(size);
                        engine.resize(size);
                    }
                    WindowEvent::RedrawRequested => {
                        if let Err(e) = engine.render() {
                            match e {
                                wgpu::SurfaceError::Lost => {
                                    let size = engine.window().inner_size();
                                    engine.resize(size);
                                }
                                wgpu::SurfaceError::OutOfMemory => event_loop.exit(),
                                wgpu::SurfaceError::Timeout | wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Other => {}
                            }
                        }
                    }
                    other => { engine.input(&other); }
                }
            }
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Continuously animate when ready
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow().as_ref() {
                engine.window().request_redraw();
            }
        });
    }

}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();

    let event_loop = winit::event_loop::EventLoop::new().map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    let mut app = App::default();
    event_loop.run_app(&mut app).map_err(|e| JsValue::from_str(&format!("{e:?}")))?;
    Ok(())
}
