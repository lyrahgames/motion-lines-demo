#version 460 core

uniform mat4 projection;
uniform mat4 view;

uniform uint transforms_count;

layout (location = 0) in vec3 p;
layout (location = 1) in vec3 n;

layout (std430, binding = 0) readonly buffer bone_weight_offsets {
  uint offsets[];
};
struct bone_weight {
  uint bid;
  float weight;
};
layout (std430, binding = 1) readonly buffer bone_weight_data {
  bone_weight weights[];
};
layout (std430, binding = 2) readonly buffer bone_transforms {
  mat4 transforms[];
};

out vec3 position;
out vec3 normal;

flat out uint id;

void main() {
  uint offset = gl_InstanceID * transforms_count;
  mat4 model = mat4(0.0);
  for (uint i = offsets[gl_VertexID]; i < offsets[gl_VertexID + 1]; ++i)
    model += weights[i].weight * transforms[offset + weights[i].bid];

  gl_Position = projection * view * model * vec4(p, 1.0);

  position = vec3(view * model * vec4(p, 1.0));
  normal = vec3(transpose(inverse(view * model)) * vec4(n, 0.0));

  id = gl_InstanceID;
}
