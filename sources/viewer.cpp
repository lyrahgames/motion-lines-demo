#include "viewer.hpp"
//
#include "log.hpp"

namespace demo {

opengl_window::opengl_window(int width, int height)
    : window(sf::VideoMode(width, height),
             "Motion Lines Demo",
             sf::Style::Default,
             sf::ContextSettings{
                 /*.depthBits = */ 24,
                 /*.stencilBits = */ 8,
                 /*.antialiasingLevel = */ 4,
                 /*.majorVersion = */ 4,
                 /*.minorVersion = */ 6,
                 /*.attributeFlags = */
                 sf::ContextSettings::Core | sf::ContextSettings::Debug,
                 /*.sRgbCapable = */ false}) {
  glbinding::initialize(sf::Context::getFunction);
  window.setVerticalSyncEnabled(true);
  window.setKeyRepeatEnabled(false);
}

viewer::viewer(int width, int height) : opengl_window{width, height} {
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_POINT_SPRITE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPointSize(10.0f);
  glLineWidth(0.5f);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  create_shader();
  create_curve_shader();

  oit_init();
  oit_create_shader();
  oit_resize(width, height);
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

  float weight = 1.0 - instance / 10.0;

  alpha = weight;

  vec4 light_color = vec4(vec3(light), alpha);

  if (instance > 0)
    light_color = light_color * colormap(weight);

  // Mix both color values.

  if (wireframe)
    frag_color = mix(line_color, light_color, mix_value);
  else
    frag_color = light_color;

  // gl_FragDepth = gl_FragCoord.z + (1.0 - gl_FragCoord.z) * instance / 10.0;


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
      }
      break;

    default:
      // do nothing
      break;
  }
}

void viewer::update() {
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

  oit_clear();

  if (not mesh.animations.empty()) {
    const auto current = std::chrono::high_resolution_clock::now();
    const auto duration =
        mesh.animations[animation].duration / mesh.animations[animation].ticks;

    if (playing) {
      time = std::fmod(
          time + std::chrono::duration<float64>(current - start).count(),
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
  contours_shader.try_set("projection", camera.projection_matrix());
  contours_shader.try_set("view", camera.view_matrix());
  contours_shader.set("line_color", vec4{vec3{0.2f}, 1.0f});
  glLineWidth(2.5f);
  contours_shader.use();
  //
  glDrawElements(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT, 0);

  const size_t trails = 10;
  {
    const float32 dt = 0.05f;
    std::vector<glm::mat4> transforms(trails * mesh.bones.size());
    load_animation_transforms(
        mesh, animation, time,
        std::span{transforms.data(), transforms.data() + mesh.bones.size()});
    for (size_t i = 1; i < trails; ++i) {
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

  // glDepthFunc(GL_LESS);
  //
  shader.try_set("projection", camera.projection_matrix());
  shader.try_set("view", camera.view_matrix());
  shader.try_set("viewport", camera.viewport_matrix());
  shader.try_set("transforms_count", mesh.bones.size());
  shader.use();
  //
  // glDrawElements(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT, 0);
  glDrawElementsInstanced(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT,
                          0, trails);

  // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  oit_shader.use();
  glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

void viewer::on_resize(int width, int height) {
  glViewport(0, 0, width, height);
  camera.set_screen_resolution(width, height);
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

  compute_motion_lines();
  compute_animation_samples();
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
  compute_motion_lines();
  compute_animation_samples();
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
}

void viewer::select_maxmin_vids() {
  select_maxmin_vids(mesh.vertices.size() * 0.01f);
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
  glGenTextures(1, &head_pointer_texture);
  //
  glGenBuffers(1, &head_pointer_initializer);
  //
  glGenBuffers(1, &atomic_counter_buffer);
  glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomic_counter_buffer);
  glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), nullptr,
               GL_DYNAMIC_COPY);
  //
  glGenBuffers(1, &fragment_storage_buffer);
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
  glBindTexture(GL_TEXTURE_2D, head_pointer_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER,
               GL_UNSIGNED_INT, nullptr);
  //
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, head_pointer_initializer);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLuint), nullptr,
               GL_STATIC_DRAW);
  GLuint* data = (GLuint*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
  std::memset(data, 0xff, width * height * sizeof(GLuint));
  glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
  //
  glBindBuffer(GL_TEXTURE_BUFFER, fragment_storage_buffer);
  glBufferData(GL_TEXTURE_BUFFER, 2 * width * height * sizeof(glm::vec4),
               nullptr, GL_DYNAMIC_COPY);

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
  // head pointer reset
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, head_pointer_initializer);
  glBindTexture(GL_TEXTURE_2D, head_pointer_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, oit_width, oit_height, 0,
               GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
  //
  glBindImageTexture(0, head_pointer_texture, 0, GL_FALSE, 0, GL_READ_WRITE,
                     GL_R32UI);
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

  for (int i = 0; i < 10; ++i){
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
  for (int i = 1; i <= 10; ++i){
    frag_color = mix(frag_color, frags[10-i].color, frags[10-i].color.a);
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

}  // namespace demo
