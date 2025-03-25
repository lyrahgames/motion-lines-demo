#version 460 core

uniform mat4 projection;
uniform mat4 view;
uniform mat4 viewport;
uniform float line_width = 10.0;
uniform float now;
uniform float delta;
uniform float char_length;

struct vertex {
  vec4 position;
  vec2 pixels;
  vec2 n;
  float time;
  float arc;
  float parc;
  float depth;
};

layout (std430, binding = 10) readonly buffer bundle_vertices {
  vertex vertices[];
};

struct segment_entry {
  uint vertex;
  uint stroke;
};

layout (std430, binding = 11) readonly buffer bundle_segments {
  segment_entry segments[];
};

layout (std430, binding = 12) readonly buffer bundle_strokes {
  float arcs[];
};

const uint elements[] = {
  0, 1, 2,
  1, 3, 2,
  2, 3, 4,
  3, 5, 4,
  4, 5, 6,
  5, 7, 6,
};

out float time;
out float v;
out float arc;

out float varc;
out float sarc;
out float speed;

flat out uint stroke;

float z_shift(uint index) {
  if ((0 < index) && (index < 3)) return 0.0;
  return 1.0;
}

void main() {
  uint sid = gl_VertexID / 18;
  uint eid = gl_VertexID % 18;
  uint element = elements[eid];
  uint vid = segments[sid].vertex + (element % 2);
  uint vid1 = segments[sid].vertex;
  uint vid2 = vid1 + 1;

  float s = float(element >> 1) - 1.5;
  v = 2.0 * s;

  vec2 x = vertices[vid].pixels;
  vec2 n = vertices[vid].n;
  float z = vertices[vid].depth;
  time = vertices[vid].time;
  arc = arcs[segments[sid].stroke] - vertices[vid].arc;
  varc = vertices[vid].arc;
  sarc = arcs[segments[sid].stroke];

  speed = (vertices[vid2].arc - vertices[vid1].arc) / (vertices[vid2].time - vertices[vid1].time);

  stroke = segments[sid].stroke;

  const float t = now - time;
  const float begin_mask = smoothstep(0.0, 0.3, arc / char_length);
  const float end_mask = 1.0 - smoothstep(0.95 * delta, delta, t);
  const float decay_mask = exp(-2.0 * t / delta);
  const float speed_value = speed * delta / char_length;
  const float dash_bound = exp(-0.2 * speed_value * speed_value);
  const float speed_mask = 1.0 - dash_bound;
  const float dash_u = mod(15.0 * varc / char_length, 2.0) - 1.0;
  // const float dash_mask = smoothstep(0.0 * dash_bound, dash_bound, abs(dash_u));
  // const float dash_mask = smoothstep(0.0, 1.0, abs(mod(10.0 * varc / char_length, 2.0) - 1.0));

  float weight = begin_mask * end_mask * decay_mask;

  vec4 r = vec4(x + s * line_width * weight * n, z + 0.0001 * z_shift(element >> 1), 1.0);

  gl_Position = vec4(inverse(viewport) * r);
}
