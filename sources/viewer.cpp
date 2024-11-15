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
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  create_shader();
  create_curve_shader();
}

void viewer::create_shader() {
  const auto vs = opengl::vertex_shader{"#version 460 core\n",  //
                                        R"##(
uniform mat4 projection;
uniform mat4 view;

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
out vec3 color;

void main() {
  // mat4 bone_transform = mat4(1.0);

  mat4 bone_transform = mat4(0.0);
  for (uint i = offsets[gl_VertexID]; i < offsets[gl_VertexID + 1]; ++i)
    bone_transform += weights[i].weight * transforms[weights[i].bid];

  gl_Position = projection * view * bone_transform * vec4(p, 1.0);

  position = vec3(view * bone_transform * vec4(p, 1.0));
  normal = vec3(transpose(inverse(view * bone_transform)) * vec4(n, 0.0));

  // color = vec3(float(offsets[gl_VertexID]) / weights.length());
  // color = vec3(float(offsets[gl_VertexID + 1] - offsets[gl_VertexID]) / 4);

  // color = vec3(0.0);
  // for (uint i = offsets[gl_VertexID]; i < offsets[gl_VertexID + 1]; ++i)
  //   color[i - offsets[gl_VertexID]] = weights[i].weight;

  // color = vec3(0.0);
  // for (uint i = offsets[gl_VertexID]; i < offsets[gl_VertexID + 1]; ++i)
  //   color[i - offsets[gl_VertexID]] = float(weights[i].bid) / transforms.length();

  color = vec3(1.0);
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
in vec3 color[];

out vec3 pos;
out vec3 nor;
out vec3 vnor;
out vec3 col;
noperspective out vec3 edge_distance;

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
  col = color[0];
  gl_Position = gl_in[0].gl_Position;
  EmitVertex();

  edge_distance = vec3(0, hb, 0);
  nor = n;
  vnor = normal[1];
  pos = position[1];
  col = color[1];
  gl_Position = gl_in[1].gl_Position;
  EmitVertex();

  edge_distance = vec3(0, 0, hc);
  nor = n;
  vnor = normal[2];
  pos = position[2];
  col = color[2];
  gl_Position = gl_in[2].gl_Position;
  EmitVertex();

  EndPrimitive();
}
)##"};

  const auto fs = opengl::fragment_shader{R"##(
#version 460 core

uniform bool wireframe = false;
uniform bool use_face_normal = false;

in vec3 pos;
in vec3 nor;
in vec3 vnor;
in vec3 col;
noperspective in vec3 edge_distance;

layout (location = 0) out vec4 frag_color;

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

  vec4 light_color = vec4(vec3(light) * col, alpha);

  // Mix both color values.

  if (wireframe)
    frag_color = mix(line_color, light_color, mix_value);
  else
    frag_color = light_color;
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
}

void viewer::create_curve_shader() {
  const auto vs = opengl::vertex_shader{R"##(
#version 460 core

uniform mat4 projection;
uniform mat4 view;

layout (location = 0) in vec3 p;

void main(){
  gl_Position = projection * view * vec4(p, 1.0);
}
)##"};

  const auto gs = opengl::geometry_shader{R"##(
#version 420 core

uniform float line_width;

uniform float screen_width;
uniform float screen_height;

layout (lines) in;
layout (triangle_strip, max_vertices = 12) out;

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
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = p + delta * n;
  uv = vec2(0.0, 1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = p - delta * n - delta * d;
  uv = vec2(-1.0, -1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = p + delta * n - delta * d;
  uv = vec2(-1.0, 1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  EndPrimitive();

  t = q - delta * n;
  uv = vec2(0.0, -1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = q + delta * n;
  uv = vec2(0.0, 1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = q - delta * n + delta * d;
  uv = vec2(1.0, -1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = q + delta * n + delta * d;
  uv = vec2(1.0, 1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  EndPrimitive();


  t = p - delta * n;
  uv = vec2(0.0, -1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = q - delta * n;
  uv = vec2(0.0, -1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos2.z, 1.0);
  EmitVertex();
  t = p + delta * n;
  uv = vec2(0.0, 1.0);
  gl_Position = vec4(2.0 * t.x / screen_width,
                     2.0 * t.y / screen_height,
                     pos1.z, 1.0);
  EmitVertex();
  t = q + delta * n;
  uv = vec2(0.0, 1.0);
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

noperspective in vec2 uv;

layout (location = 0) out vec4 frag_color;

void main() {
  // frag_color = vec4(0.1, 0.5, 0.9, 1.0);
  if (length(uv) >= 1.0) discard;
  frag_color = line_color;
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
  curve_shader.attach(gs);
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
      }
      break;

    default:
      // do nothing
      break;
  }
}

void viewer::render() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  shader.try_set("projection", camera.projection_matrix());
  shader.try_set("view", camera.view_matrix());
  shader.try_set("viewport", camera.viewport_matrix());

  shader.use();

  if (not mesh.animations.empty()) {
    const auto current = std::chrono::high_resolution_clock::now();
    const auto time =
        std::fmod(std::chrono::duration<float64>(current - start).count(),
                  mesh.animations[0].duration / mesh.animations[0].ticks);
    device.transforms.allocate_and_initialize(
        animation_transforms(mesh, 0, time));
  }

  device.va.bind();
  device.faces.bind();
  glDrawElements(GL_TRIANGLES, 3 * mesh.faces.size(), GL_UNSIGNED_INT, 0);

  // Curves
  curve_shader.use();
  curve_shader.try_set("projection", camera.projection_matrix());
  curve_shader.try_set("view", camera.view_matrix());
  curve_shader.try_set("viewport", camera.viewport_matrix());
  curve_shader.set("line_width", 3.5f);
  curve_shader.set("line_color", vec4{vec3{0.5f}, 0.8f});
  curve_shader.set("screen_width", float(camera.screen_width()));
  curve_shader.set("screen_height", float(camera.screen_height()));
  //
  curves_va.bind();
  curves_data.bind();
  for (size_t vid = 0; vid < mesh.vertices.size(); ++vid)
    glDrawArrays(GL_LINE_STRIP, vid * samples, samples);
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
      else
        turn({-0.01 * mouse_move.x, 0.01 * mouse_move.y});
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

  // camera.set_near_and_far(std::max(1e-3f * radius, radius -
  // bounding_radius),
  //                         radius + bounding_radius);
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

  constexpr float32 fps = 30.0f;
  const auto duration = mesh.animations[0].duration / mesh.animations[0].ticks;
  samples = static_cast<size_t>(std::floor(fps * duration));

  motion_lines_data.clear();
  motion_lines_data.resize(mesh.vertices.size() * samples);

  for (size_t s = 0; s < samples; ++s) {
    const auto time = s * duration / samples;
    const auto transforms = animation_transforms(mesh, 0, time);

    for (size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
      glm::mat4 x(0.0f);
      for (auto i = mesh.weights.offsets[vid];
           i < mesh.weights.offsets[vid + 1]; ++i) {
        const auto [k, weight] = mesh.weights.entries[i];
        x += weight * transforms[k];
      }
      motion_lines_data[vid * samples + s] =
          glm::vec3(x * glm::vec4(mesh.vertices[vid].position, 1.0f));
    }
  }

  curves_va.bind();
  curves_data.bind();
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
  curves_data.allocate_and_initialize(motion_lines_data);
}

}  // namespace demo
