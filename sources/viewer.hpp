#pragma once
#include "camera.hpp"
#include "scene.hpp"
#include "skinned_mesh.hpp"
//
#include "opengl/buffer.hpp"
#include "opengl/shader_program.hpp"
#include "opengl/vertex_array.hpp"
//
#include <SFML/Graphics.hpp>

namespace demo {

struct opengl_window {
  // Window and OpenGL Context
  sf::Window window{};

  opengl_window(int width, int height);
};

class viewer : public opengl_window {
  // Stores updated mouse position in window coordinates.
  sf::Vector2i mouse_pos{};

  // World Origin
  glm::vec3 origin;
  // Basis Vectors of Right-Handed Coordinate System
  glm::vec3 up{0, 1, 0};
  glm::vec3 right{1, 0, 0};
  glm::vec3 front{0, 0, 1};
  // Spherical/Horizontal Coordinates of Camera
  float radius = 10;
  float altitude = 0;
  float azimuth = 0;
  // Perspective camera
  struct camera camera {};
  bool view_should_update = false;

  //
  struct scene scene {};
  skinned_mesh mesh{};
  float bounding_radius;
  std::chrono::time_point<std::chrono::high_resolution_clock> start =
      std::chrono::high_resolution_clock::now();
  bool playing = false;
  float time{};
  int animation = 0;

  opengl::shader_program shader{};
  opengl::shader_program contours_shader{};
  opengl::shader_program curve_shader{};
  struct device_storage {
    opengl::vertex_array va{};
    opengl::vertex_buffer vertices{};
    opengl::element_buffer faces{};
    opengl::shader_storage_buffer bone_weight_offsets{};
    opengl::shader_storage_buffer bone_weight_data{};
    opengl::shader_storage_buffer transforms{};
  } device{};

  opengl::vertex_array curves_va{};
  opengl::vertex_buffer curves_data{};
  opengl::vertex_buffer curves_speed{};

  std::vector<glm::vec3> motion_lines_data{};
  std::vector<float32> motion_lines_speed{};
  size_t sample_count = 0;

  std::vector<uint32> vids{};
  bool sparse = false;

  sampled_animation samples{};
  opengl::vertex_array samples_va{};
  opengl::vertex_buffer samples_data{};

  // oit
  GLuint atomic_counter_buffer;
  int oit_width, oit_height;
  opengl::shader_program oit_shader;
  opengl::shader_storage_buffer oit_fragment_lists;
  opengl::shader_storage_buffer oit_fragment_heads;
  uint32 max_fragments;

  opengl::shader_storage_buffer ssbo_vertices{};
  opengl::shader_program ssbo_shader{};
  opengl::shader_program pvp_motion_line_shader{};

 public:
  viewer(int width = 500, int height = 500);
  void run();

  void load_scene_from_file(const std::filesystem::path& path);
  void load_vids_from_file(const std::filesystem::path& path);
  void select_maxmin_vids(size_t count);
  void select_maxmin_vids();

 private:
  void process(const sf::Event event);
  void render();
  void update();
  void update_view();

  void on_resize(int width, int height);
  void turn(const vec2& angle);
  void shift(const vec2& pixels);
  void zoom(float scale);
  // void look_at(float x, float y);
  void set_z_as_up();
  void set_y_as_up();

  void fit_view_to_surface();

  void pretty_print_node(const scene::node& node,
                         const std::string& prefix,
                         const std::string& child_prefix);
  void print_scene_info();

  void create_shader();
  void create_curve_shader();

  void compute_motion_lines();

  void select_animation(int id);

  void compute_animation_samples();

  void oit_init();
  void oit_resize(int width, int height);
  void oit_clear();
  void oit_create_shader();

  void create_ssbo_shader();
  void init_pvp();
  void create_pvp_motion_line_shader();
};

}  // namespace demo
