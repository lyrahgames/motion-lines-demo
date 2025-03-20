#include "viewer.hpp"
//
#include "log.hpp"
//
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace demo {

opengl_window::opengl_window(int width, int height)
    : window(sf::VideoMode(width, height),
             "Motion Lines Demo",
             sf::Style::Default,
             sf::ContextSettings{
                 /*.depthBits = */ 24,
                 /*.stencilBits = */ 0,
                 /*.antialiasingLevel = */ 4,
                 /*.majorVersion = */ 4,
                 /*.minorVersion = */ 6,
                 /*.attributeFlags = */
                 sf::ContextSettings::Core /*| sf::ContextSettings::Debug*/,
                 /*.sRgbCapable = */ false}) {
  glbinding::initialize(sf::Context::getFunction);
  window.setVerticalSyncEnabled(true);
  window.setKeyRepeatEnabled(false);
}

viewer::viewer(int width, int height) : opengl_window{width, height} {
  init_lua();

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  // glEnable(GL_POINT_SMOOTH);
  // glEnable(GL_POINT_SPRITE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPointSize(10.0f);
  glLineWidth(2.5f);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  create_shader();
  create_curve_shader();

  oit_init();
  oit_create_shader();
  oit_resize(width, height);

  create_ssbo_shader();
  init_pvp();
  create_pvp_motion_line_shader();

  create_surface_shader();
  create_motion_trails_shader();
  create_motion_lines_shader();

  create_bundle_shader();

  create_background_shader();
}

void viewer::create_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
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
)##"};

  const auto gs = opengl::geometry_shader{R"##(
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
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

layout (binding = 0, offset = 0) uniform atomic_uint index_counter;

struct fragment_node {
  vec4 color;
  float depth;
  float time;
  uint next;
};

layout (std430, binding = 3) writeonly buffer oit_fragment_lists {
  fragment_node fragments[];
};

layout (std430, binding = 4) coherent buffer oit_fragment_heads {
  uint heads[];
};

uniform uint max_fragments;

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

// layout (location = 0) out
vec4 frag_color;
// layout (depth_unchanged) out float gl_FragDepth;

// float colormap_red(float x) {
//     return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
// }

// float colormap_green(float x) {
//     return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
// }

// vec4 colormap(float x) {
//     float r = clamp(colormap_red(x), 0.0, 1.0);
//     float g = clamp(colormap_green(x), 0.0, 1.0);
//     float b = 1.0;
//     return vec4(r, g, b, 1.0);
// }

vec4 colormap(float x) {
    float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
    if (v > 255.0) {
        v = 510.0 - v;
    }
    v = v / 255.0;
    return vec4(v, v, v, 1.0);
}

void main() {
  if (!gl_FrontFacing) discard;

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

  float light = 0.2 + 1.0 * pow(s, 1000) + 0.75 * pow(s, 0.2);
  // float light = 0.2 + 0.75 * pow(s, 0.2);

  // Toon Shading
  if (light <= 0.50) light = 0.20;
  else if (light <= 0.60) light = 0.40;
  else if (light <= 0.80) light = 0.60;
  else if (light <= 0.90) light = 0.80;
  else if (light <= 1.00) light = 1.00;

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


  // oit
  uint index = atomicCounterIncrement(index_counter);
  if (index >= max_fragments) discard;
  uint old_head = atomicExchange(heads[width * int(gl_FragCoord.y) + int(gl_FragCoord.x)], index);


  fragment_node node;
  node.color = frag_color;
  node.depth = gl_FragCoord.z;
  node.time = float(instance);
  node.next = old_head;
  fragments[index] = node;
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!gs) {
    log::error(gs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  shader.attach(vs);
  shader.attach(gs);
  shader.attach(fs);
  shader.link();

  if (!shader.linked()) {
    log::error(shader.info_log());
    quit();
    return;
  }

  const auto contours_gs = opengl::geometry_shader{R"##(
#version 460 core

uniform mat4 projection;

layout (triangles) in;
layout (line_strip, max_vertices = 2) out;

in vec3 position[];
in vec3 normal[];

void main(){
  vec4 a = gl_in[0].gl_Position;
  vec4 b = gl_in[1].gl_Position;
  vec4 c = gl_in[2].gl_Position;

  float sa = dot(normal[0], position[0]);
  float sb = dot(normal[1], position[1]);
  float sc = dot(normal[2], position[2]);

  if (sa * sb < 0) {
    gl_Position = ((abs(sb) * a + abs(sa) * b) / (abs(sa) + abs(sb)));
    EmitVertex();
  }
  if (sa * sc < 0) {
    gl_Position = ((abs(sc) * a + abs(sa) * c) / (abs(sa) + abs(sc)));
    EmitVertex();
  }
  if (sb * sc < 0) {
    gl_Position = ((abs(sc) * b + abs(sb) * c) / (abs(sb) + abs(sc)));
    EmitVertex();
  }
  EndPrimitive();
}
)##"};

  const auto contours_fs = opengl::fragment_shader{R"##(
#version 460 core

uniform vec4 line_color;

layout (location = 0) out vec4 frag_color;
// layout (depth_unchanged) out float gl_FragDepth;

void main() {
  frag_color = line_color;
  gl_FragDepth = gl_FragCoord.z;
}
)##"};

  if (!contours_gs) {
    log::error(contours_gs.info_log());
    quit();
    return;
  }

  if (!contours_fs) {
    log::error(contours_fs.info_log());
    quit();
    return;
  }

  contours_shader.attach(vs);
  contours_shader.attach(contours_gs);
  contours_shader.attach(contours_fs);
  contours_shader.link();

  if (!contours_shader.linked()) {
    log::error(contours_shader.info_log());
    quit();
    return;
  }
}

void viewer::create_curve_shader() {
  const auto vs = opengl::vertex_shader{R"##(
#version 460 core

uniform mat4 projection;
uniform mat4 view;

layout (location = 0) in vec3 p;
layout (location = 1) in vec3 n;
layout (location = 2) in float t;
layout (location = 3) in float l;

// layout (location = 1) in float v;

// out float speed;
// out vec3 normal;

out float time;
out float length;

void main(){
  gl_Position = projection * view * vec4(p, 1.0);
  // speed = 0.0f;

  time = t;
  length = l;
}
)##"};

  const auto gs = opengl::geometry_shader{R"##(
#version 420 core

uniform float line_width;

uniform float screen_width;
uniform float screen_height;

layout (lines) in;
layout (triangle_strip, max_vertices = 12) out;

in float speed[];

out float s;
noperspective out vec2 uv;

void main(){
  // float width = 10.0;
  float width = line_width;

  vec4 pos1 = gl_in[0].gl_Position / gl_in[0].gl_Position.w;
  vec4 pos2 = gl_in[1].gl_Position / gl_in[1].gl_Position.w;

  vec2 p = vec2(0.5 * screen_width * pos1.x, 0.5 * screen_height * pos1.y);
  vec2 q = vec2(0.5 * screen_width * pos2.x, 0.5 * screen_height * pos2.y);

  vec2 d = normalize(q - p);
  vec2 n = vec2(-d.y, d.x);
  float delta = 0.5 * width;

  vec2 t = vec2(0);

  t = p - delta * n;
  uv = vec2(0.0, -1.0);
  s = speed[0];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = p + delta * n;
  uv = vec2(0.0, 1.0);
  s = speed[0];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = p - delta * n - delta * d;
  uv = vec2(-1.0, -1.0);
  s = speed[0];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = p + delta * n - delta * d;
  uv = vec2(-1.0, 1.0);
  s = speed[0];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  EndPrimitive();

  t = q - delta * n;
  uv = vec2(0.0, -1.0);
  s = speed[1];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = q + delta * n;
  uv = vec2(0.0, 1.0);
  s = speed[1];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = q - delta * n + delta * d;
  uv = vec2(1.0, -1.0);
  s = speed[1];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = q + delta * n + delta * d;
  uv = vec2(1.0, 1.0);
  s = speed[1];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  EndPrimitive();


  t = p - delta * n;
  uv = vec2(0.0, -1.0);
  s = speed[0];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = q - delta * n;
  uv = vec2(0.0, -1.0);
  s = speed[1];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = p + delta * n;
  uv = vec2(0.0, 1.0);
  s = speed[0];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = q + delta * n;
  uv = vec2(0.0, 1.0);
  s = speed[1];
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  EndPrimitive();
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

uniform vec4 line_color;

uniform float global_time;
uniform float stroke_length;
uniform float max_stroke_length;

in float time;
in float length;

// in float s;
// noperspective in vec2 uv;

layout (location = 0) out vec4 frag_color;
// layout (depth_unchanged) out float gl_FragDepth;

void main() {
  // frag_color = vec4(0.1, 0.5, 0.9, 1.0);
  // if (s < 0.2) discard;
  // if (length(uv) >= 1.0) discard;

  const float delta = 0.5;

  if (time >= global_time - 0.05) discard;
  if (time < global_time - delta) discard;

  frag_color = line_color;
  frag_color.a = (time + delta - global_time) / delta;

  // if (gl_FragCoord.z < 0.9999) discard;
  gl_FragDepth = gl_FragCoord.z;
  // gl_FragDepth = gl_FragCoord.z + 0.0001;
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!gs) {
    log::error(gs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  curve_shader.attach(vs);
  // curve_shader.attach(gs);
  curve_shader.attach(fs);
  curve_shader.link();

  if (!curve_shader.linked()) {
    log::error(curve_shader.info_log());
    quit();
    return;
  }
}

void viewer::run() {
  while (not demo::done()) {
    sf::Event event;
    while (window.pollEvent(event)) process(event);
    update();
    render();
    window.display();
  }
}

void viewer::process(const sf::Event event) {
  switch (event.type) {
    case sf::Event::Closed:
      demo::quit();
      break;

    case sf::Event::Resized:
      on_resize(event.size.width, event.size.height);
      break;

    case sf::Event::MouseWheelScrolled:
      zoom(0.1 * event.mouseWheelScroll.delta);
      break;

    case sf::Event::KeyPressed:
      switch (event.key.code) {
        case sf::Keyboard::Escape:
          demo::quit();
          break;

        case sf::Keyboard::Num1:
          set_y_as_up();
          break;
        case sf::Keyboard::Num2:
          set_z_as_up();
          break;

        case sf::Keyboard::Space:
          if (event.key.control) {
            playing = true;
            time = 0.0f;
          } else
            playing = !playing;
          break;

        case sf::Keyboard::Left:
          select_animation(--animation);
          break;

        case sf::Keyboard::Right:
          select_animation(++animation);
          break;

        case sf::Keyboard::S:
          sparse = !sparse;
          break;

        case sf::Keyboard::W:
          wireframe_rendering = !wireframe_rendering;
          surface_shader.set("wireframe", wireframe_rendering);
          break;

        case sf::Keyboard::F:
          flat_rendering = !flat_rendering;
          surface_shader.set("use_face_normal", flat_rendering);
          break;
      }
      break;

    default:
      // do nothing
      break;
  }
}

void viewer::update() {
  watcher.process();
  // process_lua_reload();
  // process_bundle_shader_reload();

  // Get new mouse position and compute movement in space.
  const auto new_mouse_pos = sf::Mouse::getPosition(window);
  const auto mouse_move = new_mouse_pos - mouse_pos;
  mouse_pos = new_mouse_pos;

  if (window.hasFocus()) {
    if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
        shift({mouse_move.x, mouse_move.y});
      else if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl))
        zoom(0.01 * mouse_move.y);
      else
        turn({-0.01 * mouse_move.x, 0.01 * mouse_move.y});
    } else if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
      playing = false;
      const auto duration = mesh.animations[animation].duration /
                            mesh.animations[animation].ticks;
      auto delta = 0.01f * mouse_move.y;
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) delta *= 0.1f;
      time = std::fmod(duration + time + delta, duration);
      device.transforms.allocate_and_initialize(
          animation_transforms(mesh, animation, time));
    }
  }

  if (view_should_update) {
    update_view();
    view_should_update = false;
  }
}

void viewer::update_view() {
  // Compute camera position by using spherical coordinates.
  // This transformation is a variation of the standard
  // called horizontal coordinates often used in astronomy.
  //
  auto p = cos(altitude) * sin(azimuth) * right +  //
           cos(altitude) * cos(azimuth) * front +  //
           sin(altitude) * up;
  p *= radius;
  p += origin;
  camera.move(p).look_at(origin, up);

  // camera.set_near_and_far(std::max(1e-3f * radius, radius - bounding_radius),
  //                         radius + bounding_radius);

  camera.set_near_and_far(
      std::max(1e-3f * bounding_radius, radius - 10.0f * bounding_radius),
      radius + 10.0f * bounding_radius);
}

void viewer::render() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Background
  //
  // glDepthFunc(GL_NEVER);
  background_shader.use();
  // glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

  // oit_clear();

  if (not mesh.animations.empty()) {
    const auto current = std::chrono::high_resolution_clock::now();
    const auto duration =
        mesh.animations[animation].duration / mesh.animations[animation].ticks;

    if (playing) {
      time = std::fmod(
          time +
              0.25f * std::chrono::duration<float64>(current - start).count(),
          duration);
      device.transforms.allocate_and_initialize(
          animation_transforms(mesh, animation, time));
    }
    start = current;

    const size_t max_samples = 35;
    const auto sample_last =
        static_cast<size_t>(std::floor(time * sample_count / duration));
    const size_t sample_first =
        (sample_last >= max_samples) ? (sample_last - max_samples) : (0);
    const auto sample_count = sample_last - sample_first;

    // curve_shader.use();
    // curve_shader.try_set("projection", camera.projection_matrix());
    // curve_shader.try_set("view", camera.view_matrix());
    // curve_shader.try_set("viewport", camera.viewport_matrix());
    // curve_shader.try_set("line_width", 1.5f);
    // // curve_shader.set("line_width", 1.5f);
    // curve_shader.try_set("line_color", vec4{vec3{0.2f}, 1.0f});
    // curve_shader.try_set("screen_width", float(camera.screen_width()));
    // curve_shader.try_set("screen_height", float(camera.screen_height()));
    // //
    // curve_shader.try_set("global_time", time);
    // curve_shader.try_set("max_stroke_length", samples.max_length);
    // //
    // samples_va.bind();
    // samples_data.bind();
    // if (sparse) {
    //   for (auto vid : vids)
    //     glDrawArrays(GL_LINE_STRIP, vid * samples.sample_count + sample_first,
    //                  sample_count);
    // } else {
    //   for (size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
    //     curve_shader.try_set(
    //         "stroke_length",
    //         samples.samples[(vid + 1) * samples.sample_count - 1].length);
    //     glDrawArrays(GL_LINE_STRIP,
    //                  vid * samples.sample_count /*+ sample_first*/,
    //                  samples.sample_count);
    //   }
    // }
  }

  // glDepthFunc(GL_ALWAYS);
  device.va.bind();
  device.faces.bind();
  //
  // contours_shader.try_set("projection", camera.projection_matrix());
  // contours_shader.try_set("view", camera.view_matrix());
  // contours_shader.set("line_color", vec4{vec3{0.2f}, 1.0f});
  // glLineWidth(2.5f);
  // contours_shader.use();
  //
  // glDrawElements(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT, 0);

  const size_t time_samples = 100;
  {
    const float32 dt = 0.01f;
    std::vector<glm::mat4> transforms(time_samples * mesh.bones.size());
    load_animation_transforms(
        mesh, animation, time,
        std::span{transforms.data(), transforms.data() + mesh.bones.size()});
    for (size_t i = 1; i < time_samples; ++i) {
      const auto t =
          std::clamp(std::floor((time - (i - 1) * dt) / dt) * dt, 0.0f, time);
      // std::clamp(time - i * dt, 0.0f, time);
      const auto first = i * mesh.bones.size();
      const auto last = (i + 1) * mesh.bones.size();
      auto view =
          std::span{transforms.data() + first, transforms.data() + last};
      load_animation_transforms(mesh, animation, t, view);
    }
    device.transforms.allocate_and_initialize(transforms);
  }

  glDepthFunc(GL_LESS);

  // shader.try_set("projection", camera.projection_matrix());
  // shader.try_set("view", camera.view_matrix());
  // shader.try_set("viewport", camera.viewport_matrix());
  // shader.try_set("transforms_count", mesh.bones.size());
  // shader.try_set("time_samples", int(time_samples));
  // shader.use();
  // //
  // // glDrawElements(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT, 0);
  // glDrawElementsInstanced(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT,
  //                         0, time_samples);

  // ssbo_shader.try_set("projection", camera.projection_matrix());
  // ssbo_shader.try_set("view", camera.view_matrix());
  // ssbo_shader.try_set("viewport", camera.viewport_matrix());
  // ssbo_shader.try_set("transforms_count", mesh.bones.size());
  // ssbo_shader.try_set("time_samples", int(time_samples));
  // ssbo_shader.try_set("width", oit_width);
  // ssbo_shader.try_set("height", oit_height);
  // ssbo_shader.try_set("max_fragments", max_fragments);
  // ssbo_shader.use();
  // //
  // glDrawElementsInstanced(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT,
  //                         0, /*time_samples*/ 1);

  // pvp_motion_line_shader.try_set("projection", camera.projection_matrix());
  // pvp_motion_line_shader.try_set("view", camera.view_matrix());
  // pvp_motion_line_shader.try_set("viewport", camera.viewport_matrix());
  // pvp_motion_line_shader.try_set("transforms_count", mesh.bones.size());
  // pvp_motion_line_shader.try_set("time_samples", int(time_samples));
  // pvp_motion_line_shader.try_set("width", oit_width);
  // pvp_motion_line_shader.try_set("height", oit_height);
  // pvp_motion_line_shader.try_set("max_fragments", max_fragments);
  // pvp_motion_line_shader.use();
  // //
  // glDrawArraysInstanced(GL_LINE_STRIP, 0, time_samples, mesh.vertices.size());

  // // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  // oit_shader.try_set("time_samples", int(time_samples));
  // oit_shader.use();
  // glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

  surface_shader.try_set("projection", camera.projection_matrix());
  surface_shader.try_set("view", camera.view_matrix());
  surface_shader.try_set("viewport", camera.viewport_matrix());
  surface_shader.try_set("transforms_count", mesh.bones.size());
  surface_shader.try_set("time_samples", int(time_samples));
  surface_shader.use();
  //
  glDrawElements(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT, 0);

  // motion_trails_shader.try_set("projection", camera.projection_matrix());
  // motion_trails_shader.try_set("view", camera.view_matrix());
  // motion_trails_shader.try_set("viewport", camera.viewport_matrix());
  // motion_trails_shader.try_set("transforms_count", mesh.bones.size());
  // motion_trails_shader.try_set("time_samples", int(time_samples));
  // motion_trails_shader.set("trails", int(10));
  // motion_trails_shader.use();
  // glDrawElementsInstanced(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT,
  //                         0, time_samples);

  // motion_lines_shader.try_set("projection", camera.projection_matrix());
  // motion_lines_shader.try_set("view", camera.view_matrix());
  // motion_lines_shader.try_set("viewport", camera.viewport_matrix());
  // motion_lines_shader.try_set("transforms_count", mesh.bones.size());
  // motion_lines_shader.try_set("time_samples", int(time_samples));
  // motion_lines_shader.try_set("line_width", 30.0f);
  // motion_lines_shader.use();
  // //
  // glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 2 * time_samples, vids.size());

  update_strokes(bundle, time, camera);
  // bundle_vertices.write(bundle.vertices);
  bundle_vertices.allocate_and_initialize(bundle.vertices);
  bundle_strokes.allocate_and_initialize(bundle.arcs);
  //
  bundle_shader.try_set("projection", camera.projection_matrix());
  bundle_shader.try_set("view", camera.view_matrix());
  bundle_shader.try_set("viewport", camera.viewport_matrix());
  // bundle_shader.try_set("line_width", 10.0f);
  bundle_shader.try_set("now", time);
  // bundle_shader.try_set("delta", 1.0f);
  bundle_shader.try_set("char_length", bounding_radius);
  bundle_shader.use();
  //
  glDrawArrays(GL_TRIANGLES, 0, 6 * segments(bundle).size());
}

void viewer::on_resize(int width, int height) {
  glViewport(0, 0, width, height);
  camera.set_screen_resolution(width, height);
  oit_resize(width, height);
  view_should_update = true;
}

void viewer::turn(const vec2& angle) {
  altitude += angle.y;
  azimuth += angle.x;
  constexpr float bound = pi / 2 - 1e-5f;
  altitude = std::clamp(altitude, -bound, bound);
  view_should_update = true;
}

void viewer::shift(const vec2& pixels) {
  const auto shift = -pixels.x * camera.right() + pixels.y * camera.up();
  const auto scale = camera.pixel_size() * radius;
  origin += scale * shift;
  view_should_update = true;
}

void viewer::zoom(float scale) {
  radius *= exp(-scale);
  view_should_update = true;
}

// void viewer::look_at(float x, float y) {
//   const auto r = primary_ray(camera, x, y);
//   if (const auto p = intersection(r, scene)) {
//     origin = r(p.t);
//     radius = p.t;
//     view_should_update = true;
//   }
// }

void viewer::set_z_as_up() {
  right = {1, 0, 0};
  front = {0, -1, 0};
  up = {0, 0, 1};
  view_should_update = true;
}

void viewer::set_y_as_up() {
  right = {1, 0, 0};
  front = {0, 0, 1};
  up = {0, 1, 0};
  view_should_update = true;
}

void viewer::pretty_print_node(const scene::node& node,
                               const std::string& prefix,
                               const std::string& child_prefix) {
  log::text(
      fmt::format("{}☋ {}: {}: {}", prefix, node.index, node.name,
                  fmt::format(fmt::fg(fmt::color::gray), "{} ◬, {} ↣",
                              node.meshes.size(), node.bone_entries.size())));

  auto property_prefix = child_prefix;
  if (node.children.empty())
    property_prefix += "  ";
  else
    property_prefix += "│ ";

  for (auto mid : node.meshes)
    log::text(fmt::format("{}{}", property_prefix,
                          fmt::format(fmt::fg(fmt::color::gray), "◬ {}: {}",
                                      mid, scene.meshes[mid].name)));

  for (auto [mid, weights] : node.bone_entries) {
    if (weights.empty()) continue;
    log::text(
        fmt::format("{}{}", property_prefix,
                    fmt::format(fmt::fg(fmt::color::gray), "↣ ◬ {}: {} ({})",
                                mid, scene.meshes[mid].name, weights.size())));
  }

  // const auto& offset = node.offset;
  // log::text(std::format("{}  {},{},{},{}", child_prefix, offset[0][0],
  //                       offset[0][1], offset[0][2], offset[0][3]));
  // log::text(std::format("{}  {},{},{},{}", child_prefix, offset[1][0],
  //                       offset[1][1], offset[1][2], offset[1][3]));
  // log::text(std::format("{}  {},{},{},{}", child_prefix, offset[2][0],
  //                       offset[2][1], offset[2][2], offset[2][3]));
  // log::text(std::format("{}  {},{},{},{}", child_prefix, offset[3][0],
  //                       offset[3][1], offset[3][2], offset[3][3]));

  auto it = node.children.begin();
  if (it == node.children.end()) return;
  auto next = it;
  ++next;
  for (; next != node.children.end(); ++next) {
    pretty_print_node(*it, child_prefix + "├─", child_prefix + "│ ");
    it = next;
  }
  pretty_print_node(*it, child_prefix + "└─", child_prefix + "  ");
}

void viewer::print_scene_info() {
  log::text("\nScene Information:");
  log::text(std::format("name = {}", scene.name));
  log::text("");

  if (not scene.meshes.empty()) {
    const auto size = scene.meshes.size();
    const auto id_width =
        static_cast<int>(std::ceil(std::log10(static_cast<float32>(size + 1))));
    log::text(fmt::format(fmt::emphasis::bold, "Meshes:"));
    for (size_t i = 0; i < scene.meshes.size() - 1; ++i) {
      const auto& mesh = scene.meshes[i];
      log::text(std::format("├─◬ {:>{}}: {}", i, id_width, mesh.name));
      // if (not mesh.name.empty()) log::text(std::format("│   {}",
      // mesh.name));
      log::text(std::format("│   #v = {:>8}", mesh.vertices.size()));
      log::text(std::format("│   #f = {:>8}", mesh.faces.size()));
    }
    {
      const auto& mesh = scene.meshes.back();
      log::text(std::format("└─◬ {:>{}}: {}", scene.meshes.size() - 1, id_width,
                            mesh.name));
      // if (not mesh.name.empty()) log::text(std::format("    {}",
      // mesh.name));
      log::text(std::format("    #v = {:>8}", mesh.vertices.size()));
      log::text(std::format("    #f = {:>8}", mesh.faces.size()));
    }
    log::text("");
  }

  log::text(fmt::format(fmt::emphasis::bold, "Hierarchy:"));
  pretty_print_node(scene.root, "", "");
  log::text("");

  log::text(fmt::format(fmt::emphasis::bold, "Animations:"));
  for (auto& anim : scene.animations) {
    log::text(std::format("  ☋ {}", anim.name));
    log::text(std::format("  time = {}", anim.duration));
    log::text(std::format("  tick = {}", anim.ticks));
    log::text("  Channels:");
    for (auto& channel : anim.channels)
      log::text(fmt::format("    {} ({},{},{})",  //
                            channel.node_name,    //
                            channel.positions.size(), channel.rotations.size(),
                            channel.scalings.size()));
    log::text("");
  }
}

void viewer::load_scene_from_file(const std::filesystem::path& path) {
  scene = scene_from_file(path);
  print_scene_info();

  mesh = skinned_mesh_from(scene);

  fit_view_to_surface();

  device.va.bind();
  device.vertices.bind();
  device.faces.bind();
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(skinned_mesh::vertex),
                        (void*)offsetof(skinned_mesh::vertex, position));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(skinned_mesh::vertex),
                        (void*)offsetof(skinned_mesh::vertex, normal));
  device.vertices.allocate_and_initialize(mesh.vertices);
  device.faces.allocate_and_initialize(mesh.faces);
  //
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, device.bone_weight_offsets.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0,
                   device.bone_weight_offsets.id());
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, device.bone_weight_data.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, device.bone_weight_data.id());
  device.bone_weight_offsets.allocate_and_initialize(mesh.weights.offsets);
  device.bone_weight_data.allocate_and_initialize(mesh.weights.entries);
  //
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, device.transforms.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, device.transforms.id());
  //
  device.transforms.allocate_and_initialize(global_transforms(mesh));

  ssbo_vertices.allocate_and_initialize(mesh.vertices);

  // compute_motion_lines();
  // compute_animation_samples();
}

void viewer::fit_view_to_surface() {
  const auto box = aabb_from(scene);
  origin = box.origin();
  bounding_radius = box.radius();

  // cout << "bounding radius = " << bounding_radius << endl;

  radius = bounding_radius / tan(0.5f * camera.vfov());
  camera.set_near_and_far(1e-5f * radius, 100 * radius);
  view_should_update = true;
}

void viewer::fit_view_to_bundle() {
  const auto box = aabb_from(bundle);
  origin = box.origin();
  bounding_radius = box.radius();

  // cout << "bounding radius = " << bounding_radius << endl;

  radius = bounding_radius / tan(0.5f * camera.vfov());
  camera.set_near_and_far(1e-5f * radius, 100 * radius);
  view_should_update = true;
}

void viewer::compute_motion_lines() {
  if (mesh.animations.empty()) return;

  constexpr float32 fps = 100.0f;
  const auto duration =
      mesh.animations[animation].duration / mesh.animations[animation].ticks;
  sample_count = static_cast<size_t>(std::floor(fps * duration));
  const auto time_step = duration / sample_count;

  motion_lines_data.clear();
  motion_lines_data.resize(mesh.vertices.size() * sample_count);

  for (size_t s = 0; s < sample_count; ++s) {
    const auto time = s * time_step;
    const auto transforms = animation_transforms(mesh, animation, time);

    for (size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
      // glm::mat4 x(0.0f);
      // for (auto i = mesh.weights.offsets[vid];
      //      i < mesh.weights.offsets[vid + 1]; ++i) {
      //   const auto [k, weight] = mesh.weights.entries[i];
      //   x += weight * transforms[k];
      // }
      const auto x = weighted_transform(mesh, transforms, vid);
      motion_lines_data[vid * sample_count + s] =
          glm::vec3(x * glm::vec4(mesh.vertices[vid].position, 1.0f));
    }
  }

  motion_lines_speed.assign(mesh.vertices.size() * sample_count, 0.0f);
  float32 max_speed = 0.0f;
  for (size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
    for (size_t s = 1; s < sample_count - 1; ++s) {
      const auto i = vid * sample_count + s;
      const auto speed =
          glm::distance(motion_lines_data[i - 1], motion_lines_data[i + 1]) /
          time_step;
      max_speed = std::max(max_speed, speed);
      motion_lines_speed[i] = speed;
    }
  }
  for (size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
    for (size_t s = 1; s < sample_count - 1; ++s) {
      const auto i = vid * sample_count + s;
      motion_lines_speed[i] /= max_speed;
    }
  }

  curves_va.bind();
  //
  curves_data.bind();
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
  curves_data.allocate_and_initialize(motion_lines_data);
  //
  curves_speed.bind();
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float32), (void*)0);
  curves_speed.allocate_and_initialize(motion_lines_speed);
}

void viewer::select_animation(int id) {
  if (mesh.animations.empty() || (id < 0)) {
    playing = false;
    animation = -1;
    device.transforms.allocate_and_initialize(global_transforms(mesh));
    return;
  }

  animation = id % mesh.animations.size();
  playing = true;
  time = 0.0f;
  // compute_motion_lines();
  // compute_animation_samples();
  compute_motion_line_bundle();
}

void viewer::load_vids_from_file(const std::filesystem::path& path) {
  std::ifstream file{path};
  uint32 vid{};
  while (file >> vid) vids.push_back(vid);
}

void viewer::select_maxmin_vids(size_t count) {
  if (mesh.vertices.empty()) return;

  vids.clear();
  vids.push_back(0);

  std::vector<float32> distances(mesh.vertices.size() * mesh.vertices.size());
  for (size_t vid1 = 0; vid1 < mesh.vertices.size(); ++vid1)
    for (size_t vid2 = 0; vid2 < mesh.vertices.size(); ++vid2)
      distances[vid1 * mesh.vertices.size() + vid2] = glm::distance(
          mesh.vertices[vid1].position, mesh.vertices[vid2].position);

  for (size_t it = 0; it < count; ++it) {
    size_t max_vid = 0;
    float32 max_distance = 0;
    for (size_t vid1 = 0; vid1 < mesh.vertices.size(); ++vid1) {
      float32 min_distance = infinity;
      for (auto vid2 : vids)
        min_distance = std::min(min_distance,
                                distances[vid1 * mesh.vertices.size() + vid2]);
      if (min_distance >= max_distance) {
        max_vid = vid1;
        max_distance = min_distance;
      }
    }
    vids.push_back(max_vid);
  }

  ssbo_vids.allocate_and_initialize(vids);
  compute_motion_line_bundle();
}

void viewer::select_maxmin_vids() {
  select_maxmin_vids(50);
}

void viewer::compute_animation_samples() {
  samples = sampled_animation_from(mesh, animation, 100);

  samples_va.bind();
  //
  samples_data.bind();
  //
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                        sizeof(sampled_animation::vertex), (void*)0);
  //
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                        sizeof(sampled_animation::vertex),
                        (void*)offsetof(sampled_animation::vertex, position));
  //
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE,
                        sizeof(sampled_animation::vertex),
                        (void*)offsetof(sampled_animation::vertex, time));
  //
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE,
                        sizeof(sampled_animation::vertex),
                        (void*)offsetof(sampled_animation::vertex, length));
  //
  samples_data.allocate_and_initialize(samples.samples);
  //
  // samples_speed.bind();
  // glEnableVertexAttribArray(1);
  // glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float32), (void*)0);
  // samples_speed.allocate_and_initialize(motion_lines_speed);
}

void viewer::oit_init() {
  glGenBuffers(1, &atomic_counter_buffer);
  glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomic_counter_buffer);
  glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), nullptr,
               GL_DYNAMIC_COPY);
  //
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, oit_fragment_lists.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, oit_fragment_lists.id());
  //
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, oit_fragment_heads.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, oit_fragment_heads.id());
}

void viewer::oit_resize(int width, int height) {
  oit_width = width;
  oit_height = height;
  //
  max_fragments = 32 * width * height;
  oit_fragment_lists.allocate(max_fragments * 2 * sizeof(glm::vec4));
  oit_fragment_heads.allocate(width * height * sizeof(GLuint));

  oit_shader.try_set("width", oit_width);
  oit_shader.try_set("height", oit_height);
  oit_shader.try_set("max_fragments", max_fragments);
  shader.try_set("width", oit_width);
  shader.try_set("height", oit_height);
  shader.try_set("max_fragments", max_fragments);
}

void viewer::oit_clear() {
  glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomic_counter_buffer);
  const GLuint zero = 0;
  glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(zero), &zero);
  //
  GLuint* data =
      (GLuint*)glMapNamedBuffer(oit_fragment_heads.id(), GL_WRITE_ONLY);
  std::memset(data, 0xff, oit_width * oit_height * sizeof(GLuint));
  glUnmapNamedBuffer(oit_fragment_heads.id());
}

void viewer::oit_create_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
const vec2 points[4] = {
  vec2(-1.0, -1.0),
  vec2(1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0)
};

void main() {
  gl_Position = vec4(points[gl_VertexID], 0.0, 1.0);
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

uniform int width;
uniform int height;
uniform int time_samples;

struct fragment_node {
  vec4 color;
  float depth;
  float time;
  uint next;
};

layout (std430, binding = 3) readonly buffer oit_fragment_lists {
  fragment_node fragments[];
};

layout (std430, binding = 4) readonly buffer oit_fragment_heads {
  uint heads[];
};

layout (location = 0) out vec4 frag_color;

float colormap_red(float x) {
    return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
}

float colormap_green(float x) {
    return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = 1.0;
    return vec4(r, g, b, 1.0);
}

fragment_node frags[256];

bool compare(fragment_node x, fragment_node y){
  if (x.time == y.time) return x.depth <= y.depth;
  return x.time < y.time;
}

void main() {
  uint index = heads[int(gl_FragCoord.y) * width + int(gl_FragCoord.x)];

  // uint count = 0;
  // while (index != 0xffffffff && count < 256){
  //   frags[count] = fragments[index];
  //   index = frags[count].next;
  //   ++count;
  // }

  // for (int i = 0; i < count; ++i){
  //   for (int j = i + 1; j < count; ++j){
  //     if (compare(frags[i], frags[j])){
  //       fragment_node tmp = frags[i];
  //       frags[i] = frags[j];
  //       frags[j] = tmp;
  //     }
  //   }
  // }

  // frag_color = vec4(0.0);
  // for (int i = 0; i < count; ++i){
  //   frag_color = mix(frag_color, frags[i].color, frags[i].color.a);
  // }

  for (int i = 0; i < time_samples; ++i){
    frags[i].color = vec4(0.0);
    frags[i].depth = 1.0;
  }

  while (index != 0xffffffff){
    fragment_node node = fragments[index];
    index = node.next;

    int t = int(node.time);
    if (node.depth < frags[t].depth)
      frags[t] = node;
  }

  frag_color = vec4(0.0);
  for (int i = 1; i <= time_samples; ++i){
    frag_color = mix(frag_color, frags[time_samples-i].color, frags[time_samples-i].color.a);
  }
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  oit_shader.attach(vs);
  oit_shader.attach(fs);
  oit_shader.link();

  if (!oit_shader.linked()) {
    log::error(oit_shader.info_log());
    quit();
    return;
  }
}

void viewer::create_ssbo_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
uniform mat4 projection;
uniform mat4 view;

uniform uint transforms_count;

struct vertex {
  float position[3];
  float normal[3];
};
layout (std430, binding = 5) readonly buffer ssbo_vertices {
  vertex vertices[];
};

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

out vec3 normal;
flat out uint instance;

void main() {
  uint offset = gl_InstanceID * transforms_count;
  mat4 model = mat4(0.0);
  for (uint i = offsets[gl_VertexID]; i < offsets[gl_VertexID + 1]; ++i)
    model += weights[i].weight * transforms[offset + weights[i].bid];

  const mat4 mv = view * model;
  const mat4 nmv = transpose(inverse(mv));

  gl_Position = projection * mv * vec4(vertices[gl_VertexID].position[0], vertices[gl_VertexID].position[1], vertices[gl_VertexID].position[2], 1.0);
  normal = vec3(nmv * vec4(vertices[gl_VertexID].normal[0], vertices[gl_VertexID].normal[1], vertices[gl_VertexID].normal[2], 0.0));
  instance = gl_InstanceID;
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

layout (binding = 0, offset = 0) uniform atomic_uint index_counter;

struct fragment_node {
  vec4 color;
  float depth;
  float time;
  uint next;
};

layout (std430, binding = 3) writeonly buffer oit_fragment_lists {
  fragment_node fragments[];
};

layout (std430, binding = 4) coherent buffer oit_fragment_heads {
  uint heads[];
};

uniform uint max_fragments;

uniform int width;
uniform int height;
uniform int time_samples;

in vec3 normal;
flat in uint instance;

vec4 colormap(float x) {
    float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
    if (v > 255.0) {
        v = 510.0 - v;
    }
    v = v / 255.0;
    return vec4(v, v, v, 1.0);
}

void main() {
  if (!gl_FrontFacing) discard;

  const float s = abs(normalize(normal).z);
  float light = 0.2 + 1.0 * pow(s, 1000) + 0.75 * pow(s, 0.2);
  //
  if (light <= 0.50) light = 0.20;
  else if (light <= 0.60) light = 0.40;
  else if (light <= 0.80) light = 0.60;
  else if (light <= 0.90) light = 0.80;
  else if (light <= 1.00) light = 1.00;

  const float weight = 1.0 - instance / float(time_samples);
  vec4 color = vec4(vec3(light), weight);
  if (instance > 0)
    color = color * colormap(weight);

  const uint index = atomicCounterIncrement(index_counter);
  if (index >= max_fragments) discard;
  uint head = atomicExchange(heads[width * int(gl_FragCoord.y) + int(gl_FragCoord.x)], index);
  fragment_node node;
  node.color = color;
  node.depth = gl_FragCoord.z;
  node.time = float(instance);
  node.next = head;
  fragments[index] = node;
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  ssbo_shader.attach(vs);
  ssbo_shader.attach(fs);
  ssbo_shader.link();

  if (!ssbo_shader.linked()) {
    log::error(ssbo_shader.info_log());
    quit();
    return;
  }
}

void viewer::init_pvp() {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_vertices.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo_vertices.id());
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_vids.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo_vids.id());
}

void viewer::create_pvp_motion_line_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
uniform mat4 projection;
uniform mat4 view;

uniform uint transforms_count;

struct vertex {
  float position[3];
  float normal[3];
};
layout (std430, binding = 5) readonly buffer ssbo_vertices {
  vertex vertices[];
};

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

out vec3 normal;
flat out uint instance;

void main() {
  uint offset = gl_VertexID * transforms_count;
  mat4 model = mat4(0.0);
  for (uint i = offsets[gl_InstanceID]; i < offsets[gl_InstanceID + 1]; ++i)
    model += weights[i].weight * transforms[offset + weights[i].bid];

  const mat4 mv = view * model;
  const mat4 nmv = transpose(inverse(mv));

  gl_Position = projection * mv * vec4(vertices[gl_InstanceID].position[0], vertices[gl_InstanceID].position[1], vertices[gl_InstanceID].position[2], 1.0);
  normal = vec3(nmv * vec4(vertices[gl_InstanceID].normal[0], vertices[gl_InstanceID].normal[1], vertices[gl_InstanceID].normal[2], 0.0));
  instance = gl_VertexID;
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

layout (binding = 0, offset = 0) uniform atomic_uint index_counter;

struct fragment_node {
  vec4 color;
  float depth;
  float time;
  uint next;
};

layout (std430, binding = 3) writeonly buffer oit_fragment_lists {
  fragment_node fragments[];
};

layout (std430, binding = 4) coherent buffer oit_fragment_heads {
  uint heads[];
};

uniform uint max_fragments;

uniform int width;
uniform int height;
uniform int time_samples;

in vec3 normal;
flat in uint instance;

// float colormap_red(float x) {
//     return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
// }

// float colormap_green(float x) {
//     return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
// }

// vec4 colormap(float x) {
//     float r = clamp(colormap_red(x), 0.0, 1.0);
//     float g = clamp(colormap_green(x), 0.0, 1.0);
//     float b = 1.0;
//     return vec4(r, g, b, 1.0);
// }

vec4 colormap(float x) {
    float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
    if (v > 255.0) {
        v = 510.0 - v;
    }
    v = v / 255.0;
    return vec4(v, v, v, 1.0);
}

void main() {
  if (!gl_FrontFacing) discard;

  // const float s = abs(normalize(normal).z);
  // float light = 0.2 + 1.0 * pow(s, 1000) + 0.75 * pow(s, 0.2);
  //
  // if (light <= 0.50) light = 0.20;
  // else if (light <= 0.60) light = 0.40;
  // else if (light <= 0.80) light = 0.60;
  // else if (light <= 0.90) light = 0.80;
  // else if (light <= 1.00) light = 1.00;

  const float weight = 1.0 - instance / float(time_samples);
  vec4 color = vec4(vec3(0.2), weight);
  if (instance > 0)
    color = color * colormap(weight);

  const uint index = atomicCounterIncrement(index_counter);
  if (index >= max_fragments) discard;
  uint head = atomicExchange(heads[width * int(gl_FragCoord.y) + int(gl_FragCoord.x)], index);
  fragment_node node;
  node.color = color;
  node.depth = gl_FragCoord.z;
  node.time = 1;
  node.next = head;
  fragments[index] = node;
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  pvp_motion_line_shader.attach(vs);
  pvp_motion_line_shader.attach(fs);
  pvp_motion_line_shader.link();

  if (!pvp_motion_line_shader.linked()) {
    log::error(pvp_motion_line_shader.info_log());
    quit();
    return;
  }
}

void viewer::create_surface_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
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
)##"};

  const auto gs = opengl::geometry_shader{R"##(
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
)##"};

  const auto fs = opengl::fragment_shader{R"##(
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

  float light = 0.2 + 1.0 * pow(s, 1000) + 0.75 * pow(s, 0.2);
  // float light = 0.2 + 0.75 * pow(s, 0.2);

  // Toon Shading
  // if (light <= 0.50) light = 0.20;
  // else if (light <= 0.60) light = 0.40;
  // else if (light <= 0.80) light = 0.60;
  // else if (light <= 0.90) light = 0.80;
  // else if (light <= 1.00) light = 1.00;

  if (light <= 0.85)
    light = 0.5;
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
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!gs) {
    log::error(gs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  surface_shader.attach(vs);
  surface_shader.attach(gs);
  surface_shader.attach(fs);
  surface_shader.link();

  if (!surface_shader.linked()) {
    log::error(surface_shader.info_log());
    quit();
    return;
  }
}

void viewer::create_motion_trails_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
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
  uint offset = (gl_InstanceID + 1) * transforms_count;
  mat4 model = mat4(0.0);
  for (uint i = offsets[gl_VertexID]; i < offsets[gl_VertexID + 1]; ++i)
    model += weights[i].weight * transforms[offset + weights[i].bid];

  gl_Position = projection * view * model * vec4(p, 1.0);

  position = vec3(view * model * vec4(p, 1.0));
  normal = vec3(transpose(inverse(view * model)) * vec4(n, 0.0));

  id = gl_InstanceID + 1;
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

uniform int width;
uniform int height;
uniform int time_samples;
uniform int trails;

in vec3 position;
in vec3 normal;
flat in uint id;

layout (location = 0) out vec4 frag_color;

// vec4 colormap(float x) {
//     float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
//     if (v > 255.0) {
//         v = 510.0 - v;
//     }
//     v = v / 255.0;
//     return vec4(v, v, v, 1.0);
// }

float colormap_red(float x) {
    return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
}

float colormap_green(float x) {
    return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = 1.0;
    return vec4(r, g, b, 1.0);
}

void main() {
  if (bool(id % ((time_samples - 1) / trails))) discard;

  float s = abs(normalize(normal).z);
  float light = 0.2 + 1.0 * pow(s, 1000) + 0.75 * pow(s, 0.2);
  // Toon Shading
  if (light <= 0.50) light = 0.20;
  else if (light <= 0.60) light = 0.40;
  else if (light <= 0.80) light = 0.60;
  else if (light <= 0.90) light = 0.80;
  else if (light <= 1.00) light = 1.00;

  float weight = 1.0 - id / float(time_samples);
  vec4 light_color = vec4(vec3(light), 1.0);
  light_color = light_color * colormap(weight);
  frag_color = light_color;
  gl_FragDepth = gl_FragCoord.z + (1.0 - gl_FragCoord.z) * id / time_samples;
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  motion_trails_shader.attach(vs);
  motion_trails_shader.attach(fs);
  motion_trails_shader.link();

  if (!motion_trails_shader.linked()) {
    log::error(motion_trails_shader.info_log());
    quit();
    return;
  }
}

void viewer::create_motion_lines_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
uniform mat4 projection;
uniform mat4 view;
uniform mat4 viewport;
uniform uint transforms_count;
uniform int time_samples;
uniform float line_width;

struct vertex {
  float position[3];
  float normal[3];
};
layout (std430, binding = 5) readonly buffer ssbo_vertices {
  vertex vertices[];
};

layout (std430, binding = 6) readonly buffer ssbo_vids {
  uint vids[];
};

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

out float time;
out float v;

vec4 vertex_position(uint vid) {
  return vec4(vertices[vid].position[0], vertices[vid].position[1], vertices[vid].position[2], 1.0);
}

mat4 animated_model_transform(uint tid, uint vid) {
  uint offset = tid * transforms_count;
  mat4 model = mat4(0.0);
  for (uint i = offsets[vid]; i < offsets[vid + 1]; ++i)
    model += weights[i].weight * transforms[offset + weights[i].bid];
  return model;
}

vec4 screenspace_position(vec4 x) {
  const vec4 pos = projection * view * x;
  const vec4 tmp = pos / pos.w;
  return viewport * tmp;
}

void main() {
  const uint vid = vids[gl_InstanceID];
  // const uint vid = gl_InstanceID;

  const uint tid = gl_VertexID >> 1;
  time = float(tid);
  const float sgn = bool(gl_VertexID & 1) ? -1.0 : 1.0;

  const vec4 pos = vertex_position(vid);

  const vec2 p = vec2(screenspace_position(animated_model_transform(tid - 1, vid) * pos));
  const vec4 _x = screenspace_position(animated_model_transform(tid + 0, vid) * pos);
  const vec2 x = vec2(_x);
  const vec2 q = vec2(screenspace_position(animated_model_transform(tid + 1, vid) * pos));

  const vec2 xp = normalize(p - x);
  const vec2 xq = normalize(q - x);

  // Tangent
  vec2 t = normalize(xq - xp);
  if (tid == 0)
    t = normalize(xq);
  else if (tid == time_samples - 1)
    t = -normalize(xp);
  // Normal
  const vec2 n = vec2(-t.y, t.x);

  const vec4 r = vec4(x + 0.5 * line_width * sgn * n, _x.z, 1.0);

  gl_Position = vec4(inverse(viewport) * r);
  v = sgn;
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

uniform int time_samples;

in float time;
in float v;

layout (location = 0) out vec4 frag_color;

float colormap_red(float x) {
    return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
}

float colormap_green(float x) {
    return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = 1.0;
    return vec4(r, g, b, 1.0);
}

// vec4 colormap(float x) {
//     float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
//     if (v > 255.0) {
//         v = 510.0 - v;
//     }
//     v = v / 255.0;
//     return vec4(v, v, v, 1.0);
// }

void main() {
  const float t = time / float(time_samples);
  const float weight = 100.0 * t * t * exp(-10.0 * t);
  if ((v > weight * sin(100.0 * t) + weight) || (v < weight * sin(100.0 * t) - weight)) discard;
  frag_color = colormap(weight) * vec4(vec3(1.0), weight);
  // gl_FragDepth = gl_FragCoord.z + (1.0 - gl_FragCoord.z) * time / time_samples;
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  motion_lines_shader.attach(vs);
  motion_lines_shader.attach(fs);
  motion_lines_shader.link();

  if (!motion_lines_shader.linked()) {
    log::error(motion_lines_shader.info_log());
    quit();
    return;
  }
}

void viewer::compute_motion_line_bundle() {
  bundle = uniform_motion_line_bundle(mesh, vids, animation, 1000);
  fit_view_to_bundle();

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bundle_vertices.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, bundle_vertices.id());
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bundle_segments.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, bundle_segments.id());
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bundle_strokes.id());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, bundle_strokes.id());
  bundle_vertices.allocate_and_initialize(bundle.vertices);
  bundle_segments.allocate_and_initialize(segments(bundle));
  bundle_strokes.allocate_and_initialize(bundle.arcs);
}

void viewer::create_bundle_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
uniform mat4 projection;
uniform mat4 view;
uniform mat4 viewport;
uniform float line_width = 1.0;

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
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

uniform float now = 0.0;
uniform float delta = 1.0;
uniform float char_length = 1.0;

in float time;
in float arc;
in float v;
in float varc;
in float sarc;
in float speed;

layout (location = 0) out vec4 frag_color;

// float colormap_red(float x) {
//     return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
// }

// float colormap_green(float x) {
//     return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
// }

// vec4 colormap(float x) {
//     float r = clamp(colormap_red(x), 0.0, 1.0);
//     float g = clamp(colormap_green(x), 0.0, 1.0);
//     float b = 1.0;
//     return vec4(r, g, b, 1.0);
// }

// vec4 colormap(float x) {
//     float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
//     if (v > 255.0) {
//         v = 510.0 - v;
//     }
//     v = v / 255.0;
//     return vec4(v, v, v, 1.0);
// }

// float colormap_f(float x) {
//   if (x < 0.8110263645648956) {
//     return (((4.41347880412638E+03 * x - 1.18250308887283E+04) * x + 1.13092070303101E+04) * x - 4.94879610401395E+03) * x + 1.10376673162241E+03;
//   } else {
//     return (4.44045986053970E+02 * x - 1.34196160353499E+03) * x + 9.26518306556645E+02;
//   }
// }

// float colormap_red(float x) {
//   if (x < 0.09384074807167053) {
//     return 7.56664615384615E+02 * x + 1.05870769230769E+02;
//   } else if (x < 0.3011957705020905) {
//     return (-2.97052932130813E+02 * x + 4.43575866219751E+02) * x + 1.37867123966178E+02;
//   } else if (x < 0.3963058760920129) {
//     return 8.61868131868288E+01 * x + 2.18562881562874E+02;
//   } else if (x < 0.5) {
//     return 2.19915384615048E+01 * x + 2.44003846153861E+02;
//   } else {
//     return colormap_f(x);
//   }
// }

// float colormap_green(float x) {
//   if (x < 0.09568486400411116) {
//     return 2.40631111111111E+02 * x + 1.26495726495727E+00;
//   } else if (x < 0.2945883673263987) {
//     return 7.00971783488427E+02 * x - 4.27826773670273E+01;
//   } else if (x < 0.3971604611945229) {
//     return 5.31775726495706E+02 * x + 7.06051282052287E+00;
//   } else if (x < 0.5) {
//     return 3.64925470085438E+02 * x + 7.33268376068493E+01;
//   } else {
//     return colormap_f(x);
//   }
// }

// float colormap_blue(float x) {
//   if (x < 0.09892375498249567) {
//     return 1.30670329670329E+02 * x + 3.12116402116402E+01;
//   } else if (x < 0.1985468629735229) {
//     return 3.33268034188035E+02 * x + 1.11699145299146E+01;
//   } else if (x < 0.2928770209555256) {
//     return 5.36891330891336E+02 * x - 2.92588522588527E+01;
//   } else if (x < 0.4061551302245808) {
//     return 6.60915763546766E+02 * x - 6.55827586206742E+01;
//   } else if (x < 0.5) {
//     return 5.64285714285700E+02 * x - 2.63359683794383E+01;
//   } else {
//     return colormap_f(x);
//   }
// }

// vec4 colormap(float x) {
//   float r = clamp(colormap_red(x) / 255.0, 0.0, 1.0);
//   float g = clamp(colormap_green(x) / 255.0, 0.0, 1.0);
//   float b = clamp(colormap_blue(x) / 255.0, 0.0, 1.0);
//   return vec4(r, g, b, 1.0);
// }

vec4 colormap(float x) {
    float r = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    float g = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    float b = clamp(4.0 * x - 3.0, 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main() {
  const float t = now - time;
  if ((t < 0.0) || (t > delta)) discard;

  const float begin_mask = smoothstep(0.01, 0.02, arc / char_length);
  const float end_mask = 1.0 - smoothstep(0.95 * delta, delta, t);
  const float decay_mask = exp(-2.0 * t / delta);
  const float speed_value = speed * delta / char_length;
  const float dash_bound = exp(-0.15 * speed_value * speed_value);
  const float speed_mask = 1.0 - dash_bound;
  const float dash_u = mod(20.0 * varc / char_length, 2.0) - 1.0;
  const float dash_mask = smoothstep(0.95 * dash_bound, dash_bound, abs(dash_u));
  // const float dash_mask = smoothstep(0.20, 0.30, abs(mod(50.0 * varc / char_length, 2.0) - 1.0));

  float weight = begin_mask * end_mask * decay_mask * speed_mask;

  const float width_mask = (1.0 - smoothstep(0.95 * weight, weight, abs(v)));
  weight *= width_mask;

  // frag_color = vec4(vec3(0.5), weight);
  frag_color = vec4(vec3(colormap(0.1 + t / delta / 0.9)), weight);


  // const float l = arc / 500.0;
  // const float speed = arc / t;
  // const float weight = 20.0 * l * l * exp(-10.0 * t);
  // if ((v > weight * sin(100.0 * l) + weight) || (v < weight * sin(100.0 * l) - weight)) discard;
  // float weight = smoothstep(5.0, 10.0, arc) * exp(-4.0 * t) * smoothstep(500.0, 501.0, speed);
  // float valpha = 1.0 - smoothstep(0.4, 0.9, abs(v));
  // float dash = smoothstep(0.28, 0.32, abs(mod(varc / 4.0, 2.0) - 1.0));
  // weight = valpha * weight;
  // frag_color = mix(vec4(1.0), vec4(vec3(0.2), 1.0), weight);
  // frag_color = vec4(vec3(0.2), valpha * weight * dash);
  // frag_color = mix(colormap(weight) * vec4(vec3(1.0), 1.0), vec4(vec3(1.0), weight), weight);
  // gl_FragDepth = gl_FragCoord.z + (1.0 - gl_FragCoord.z) * t / time_samples;
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  bundle_shader.attach(vs);
  bundle_shader.attach(fs);
  bundle_shader.link();

  if (!bundle_shader.linked()) {
    log::error(bundle_shader.info_log());
    quit();
    return;
  }
}

void viewer::load_background_from_file(const std::filesystem::path& path) {
  int width, height, nrChannels;
  unsigned char* data =
      stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
  if (!data) {
    log::error("Failed to load background texture!");
    return;
  }
  glBindTexture(GL_TEXTURE_2D, background_texture);
  if (nrChannels == 3)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, data);
  else if (nrChannels == 4)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, data);

  glGenerateMipmap(GL_TEXTURE_2D);

  stbi_image_free(data);

  log::info("Successfully loaded background texture!");
}

void viewer::create_background_shader() {
  glGenTextures(1, &background_texture);
  glBindTexture(GL_TEXTURE_2D, background_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
const vec2 points[4] = {
  vec2(-1.0, -1.0),
  vec2(1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0)
};

out vec2 uv;

void main() {
  uv = 0.5 * (points[gl_VertexID] + 1.0);
  gl_Position = vec4(points[gl_VertexID], 0.99999, 1.0);
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

uniform sampler2D background;

in vec2 uv;

layout (location = 0) out vec4 frag_color;

void main() {
  frag_color = texture(background, uv);
}
)##"};

  if (!vs) {
    log::error(vs.info_log());
    quit();
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    quit();
    return;
  }

  background_shader.attach(vs);
  background_shader.attach(fs);
  background_shader.link();

  if (!background_shader.linked()) {
    log::error(background_shader.info_log());
    quit();
    return;
  }
}

namespace {
using namespace xstd;

template <meta::string str, meta::string docstr, typename functor>
struct function_entry : functor {
  using base = functor;
  using base::operator();
  static consteval auto name() noexcept { return str; }
  static consteval auto docstring() noexcept { return docstr; }
};

template <meta::string name, meta::string docstr>
constexpr auto fn(auto&& f) noexcept {
  return function_entry<name, docstr, std::unwrap_ref_decay_t<decltype(f)>>{
      std::forward<decltype(f)>(f)};
}

template <typename type>
concept function_entry_instance =
    matches<type,
            []<meta::string name, meta::string docstr, typename functor>(
                function_entry<name, docstr, functor>) {
              return meta::as_signature<true>;
            }>;

template <typename... types>
struct function_list : std::tuple<types...> {
  using base = std::tuple<types...>;
  using base::base;
};

template <typename... types>
constexpr auto function_list_from(types... values) {
  return named_tuple<meta::name_list<types::name()...>, std::tuple<types...>>{
      values...};
}
}  // namespace

void viewer::init_lua() {
  lua.open_libraries(  //
      sol::lib::base, sol::lib::package, sol::lib::coroutine, sol::lib::string,
      sol::lib::os, sol::lib::math, sol::lib::table, sol::lib::debug,
      sol::lib::bit32, sol::lib::io);

  static auto functions = std::tuple{
      fn<"quit", "Quit the application.">([] { quit(); }),

      fn<"load_scene_from_file", "Load animated scene from given file path.">(
          [this](czstring path) { load_scene_from_file(path); }),

      fn<"create_seeds", "Compute a given number of motion line seeds.">(
          [this](int n) { select_maxmin_vids(n); }),

      fn<"load_surface_shader", "Load all shader files for surface rendering.">(
          [this](const string& vpath, const string& gpath,
                 const string& fpath) {
            load_surface_shader(vpath, gpath, fpath);
          }),

      fn<"load_bundle_shader",
         "Load vertex and fragment shader files for motion line shader.">(
          [this](const string& vpath, const string& fpath) {
            load_bundle_shader(vpath, fpath);
          }),
  };

  for_each(functions, [this](auto& f) {
    using entry = std::decay_t<decltype(f)>;
    lua.set_function(view_from(entry::name()), f);
  });

  lua.set_function("help", [this] {
    string out{};
    for_each(functions, [this, &out](auto& f) {
      using entry = std::decay_t<decltype(f)>;
      out += std::format("{}\n{}\n\n", entry::name(), entry::docstring());
    });
    log::info(out);
  });
}

void viewer::eval_lua_file(const std::filesystem::path& path) {
  const auto cwd = std::filesystem::current_path();
  current_path(path.parent_path());
  lua.safe_script_file(path);
  current_path(cwd);

  watcher.watch([this](auto& path) { eval_lua_file(path); }, path);
}

void viewer::load_surface_shader(const std::filesystem::path& vpath,
                                 const std::filesystem::path& gpath,
                                 const std::filesystem::path& fpath) {
  const auto vs = opengl::vertex_shader{xstd::string_from_file(vpath)};
  const auto gs = opengl::geometry_shader{xstd::string_from_file(gpath)};
  const auto fs = opengl::fragment_shader{xstd::string_from_file(fpath)};

  if (!vs) {
    log::error(vs.info_log());
    return;
  }

  if (!gs) {
    log::error(gs.info_log());
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    return;
  }

  opengl::shader_program tmp{};

  tmp.attach(vs);
  tmp.attach(gs);
  tmp.attach(fs);
  tmp.link();

  if (!tmp.linked()) {
    log::error(tmp.info_log());
    return;
  }

  surface_shader = move(tmp);

  watcher.watch(
      [this](auto& vpath, auto& gpath, auto& fpath) {
        load_surface_shader(vpath, gpath, fpath);
      },
      vpath, gpath, fpath);
}

void viewer::load_bundle_shader(const std::filesystem::path& vpath,
                                const std::filesystem::path& fpath) {
  const auto vs = opengl::vertex_shader{xstd::string_from_file(vpath)};
  const auto fs = opengl::fragment_shader{xstd::string_from_file(fpath)};

  if (!vs) {
    log::error(vs.info_log());
    return;
  }

  if (!fs) {
    log::error(fs.info_log());
    return;
  }

  opengl::shader_program tmp{};

  tmp.attach(vs);
  tmp.attach(fs);
  tmp.link();

  if (!tmp.linked()) {
    log::error(tmp.info_log());
    return;
  }

  bundle_shader = move(tmp);

  watcher.watch(
      [this](auto& vpath, auto& fpath) { load_bundle_shader(vpath, fpath); },
      vpath, fpath);
}

}  // namespace demo
