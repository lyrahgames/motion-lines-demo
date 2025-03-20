#version 460 core

uniform mat4 projection;
uniform mat4 view;
uniform mat4 viewport;
uniform float line_width = 10.0;

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
};

out float time;
out float v;
out float arc;

out float varc;
out float sarc;
out float speed;

void main() {
  uint sid = gl_VertexID / 6;
  uint eid = gl_VertexID % 6;
  uint element = elements[eid];
  uint vid = segments[sid].vertex + (element % 2);
  uint vid1 = segments[sid].vertex;
  uint vid2 = vid1 + 1;

  float s = float(element >> 1) - 0.5;
  v = 2.0 * s;

  vec2 x = vertices[vid].pixels;
  vec2 n = vertices[vid].n;
  float z = vertices[vid].depth;
  time = vertices[vid].time;
  arc = arcs[segments[sid].stroke] - vertices[vid].arc;
  varc = vertices[vid].arc;
  sarc = arcs[segments[sid].stroke];

  speed = (vertices[vid2].arc - vertices[vid1].arc) / (vertices[vid2].time - vertices[vid1].time);

  vec4 r = vec4(x + s * line_width * n, z, 1.0);

  gl_Position = vec4(inverse(viewport) * r);
}
