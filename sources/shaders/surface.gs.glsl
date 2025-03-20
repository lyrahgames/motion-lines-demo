#version 460 core

uniform mat4 view;
uniform mat4 viewport;

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 position[];
in vec3 normal[];
flat in uint id[];

out vec3 pos;
out vec3 nor;
out vec3 vnor;
noperspective out vec3 edge_distance;
flat out uint instance;

void main(){
  vec3 p0 = vec3(viewport * (gl_in[0].gl_Position /
                             gl_in[0].gl_Position.w));
  vec3 p1 = vec3(viewport * (gl_in[1].gl_Position /
                             gl_in[1].gl_Position.w));
  vec3 p2 = vec3(viewport * (gl_in[2].gl_Position /
                             gl_in[2].gl_Position.w));

  float a = length(p1 - p2);
  float b = length(p2 - p0);
  float c = length(p1 - p0);

  vec3 n = normalize(cross(gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz, gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz));

  float alpha = acos((b * b + c * c - a * a) / (2.0 * b * c));
  float beta  = acos((a * a + c * c - b * b) / (2.0 * a * c));

  float ha = abs(c * sin(beta));
  float hb = abs(c * sin(alpha));
  float hc = abs(b * sin(alpha));

  gl_PrimitiveID = gl_PrimitiveIDIn;

  edge_distance = vec3(ha, 0, 0);
  nor = n;
  vnor = normal[0];
  pos = position[0];
  gl_Position = gl_in[0].gl_Position;
  instance = id[0];
  EmitVertex();

  edge_distance = vec3(0, hb, 0);
  nor = n;
  vnor = normal[1];
  pos = position[1];
  gl_Position = gl_in[1].gl_Position;
  instance = id[1];
  EmitVertex();

  edge_distance = vec3(0, 0, hc);
  nor = n;
  vnor = normal[2];
  pos = position[2];
  gl_Position = gl_in[2].gl_Position;
  instance = id[2];
  EmitVertex();

  EndPrimitive();
}
