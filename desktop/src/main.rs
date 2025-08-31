use anyhow::Result;
use engine::EngineApp;
use winit::application::ApplicationHandler;
use winit::event::{WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

struct App {
    engine: Option<EngineApp>,
    window_id: Option<WindowId>,
}

impl Default for App {
    fn default() -> Self { Self { engine: None, window_id: None } }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.engine.is_none() {
            let attrs = Window::default_attributes().with_title("Demo Engine (Desktop)");
            let window = event_loop.create_window(attrs).expect("create window");
            self.window_id = Some(window.id());
            let engine = pollster::block_on(EngineApp::new(window)).expect("engine init");
            self.engine = Some(engine);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let Some(engine) = self.engine.as_mut() else { return; };
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                engine.resize(new_size);
            }
            WindowEvent::ScaleFactorChanged { scale_factor: _, mut inner_size_writer } => {
                // Request maintaining the current physical size so swapchain tracks DPI
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
                        wgpu::SurfaceError::Timeout | wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Other => {
                            // ignore
                        }
                    }
                }
            }
            other => {
                engine.input(&other);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(engine) = &self.engine { engine.window().request_redraw(); }
    }
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
