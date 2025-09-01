struct Uniforms {
  mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VSIn {
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>, // present to match pipeline layout; ignored
};

struct VSOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(input: VSIn) -> VSOut {
  var out: VSOut;
  out.pos = uniforms.mvp * vec4<f32>(input.position, 1.0);
  return out;
}

@fragment
fn fs_main(_in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
