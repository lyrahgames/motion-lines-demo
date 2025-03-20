#version 460 core

uniform int width;
uniform int height;
uniform int time_samples;

uniform bool wireframe = false;
uniform bool use_face_normal = false;

in vec3 pos;
in vec3 nor;
in vec3 vnor;
noperspective in vec3 edge_distance;
flat in uint instance;

layout (location = 0) out vec4 frag_color;

vec4 colormap(float x) {
    float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
    if (v > 255.0) {
        v = 510.0 - v;
    }
    v = v / 255.0;
    return vec4(v, v, v, 1.0);
}

void main() {

  // Compute distance from edges.
  float d = min(edge_distance.x, edge_distance.y);
  d = min(d, edge_distance.z);
  float line_width = 0.01;
  float line_delta = 1.0;
  float alpha = 1.0;
  vec4 line_color = vec4(vec3(0.5), alpha);

  float mix_value =
      smoothstep(line_width - line_delta, line_width + line_delta, d);

  // float mix_value = 1.0;
  // Compute viewer shading.

  float s = abs(normalize(vnor).z);
  if (use_face_normal)
    s = abs(normalize(nor).z);

  // float light = 0.2 + 1.0 * pow(s, 1000) + 0.75 * pow(s, 0.2);
  // float light = 0.2 + 0.75 * pow(s, 0.2);

  // Toon Shading
  // if (light <= 0.50) light = 0.20;
  // else if (light <= 0.60) light = 0.40;
  // else if (light <= 0.80) light = 0.60;
  // else if (light <= 0.90) light = 0.80;
  // else if (light <= 1.00) light = 1.00;

  float light = pow(s, 0.5);
  if (light <= 0.55)
    light = 0.2;
  else if (light <= 0.8)
    light = 0.8;
  else
    light = 1.0;

  float weight = 1.0 - instance / float(time_samples);

  alpha = weight;

  vec4 light_color = vec4(vec3(light), alpha);

  if (instance > 0)
    light_color = light_color * colormap(weight);

  // Mix both color values.

  if (wireframe)
    frag_color = mix(line_color, light_color, mix_value);
  else
    frag_color = light_color;

  // gl_FragDepth = gl_FragCoord.z + (1.0 - gl_FragCoord.z) * instance / time_samples;
}
