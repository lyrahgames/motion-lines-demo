#pragma once
#include "defaults.hpp"

namespace demo::opengl {

struct shader_object_handle : object_handle {
  using base = object_handle;
  using base::base;

  bool valid() const noexcept { return glIsShader(handle) == GL_TRUE; }

  bool compiled() const noexcept {
    // 'glGetShaderiv' will generate errors 'GL_INVALID_VALUE' or
    // 'GL_INVALID_OPERATION' if the shader handle is not valid.
    // If an error is generated, no change is made to 'status'.
    // Also, newly generated shader objects are not compiled.
    // So, 'status' can only be 'GL_TRUE'
    // if the shader is valid and compiled.
    //
    auto status = static_cast<GLint>(GL_FALSE);
    glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
    return static_cast<GLboolean>(status) == GL_TRUE;
  }

  explicit operator bool() const noexcept { return compiled(); }

  auto info_log() const -> string {
    GLint info_log_size;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &info_log_size);
    string info_log{};
    info_log.resize(info_log_size);
    glGetShaderInfoLog(handle, info_log_size, nullptr, info_log.data());
    return info_log;
  }

  using GLstring = const GLchar*;
  static auto data(czstring str) noexcept -> GLstring { return str; }
  static auto size(czstring str) noexcept -> GLint { return -1; }
  static auto data(string_view str) noexcept -> GLstring { return str.data(); }
  static auto size(string_view str) noexcept -> GLint { return str.size(); }

  void set_source(auto&&... str) noexcept {
    // OpenGL copies the shader source code strings
    // when glShaderSource is called, so an application
    // may free its copy of the source code strings
    // immediately after the function returns.
    //
    constexpr GLsizei count = sizeof...(str);
    array<GLstring, count> strings{data(str)...};
    array<GLint, count> lengths{size(str)...};
    glShaderSource(handle, count, strings.data(), lengths.data());
  }

  void compile() noexcept { glCompileShader(handle); }

  bool compile(auto&&... str) noexcept {
    set_source(forward<decltype(str)>(str)...);
    compile();
    return compiled();
  }
};

constexpr auto shader_object_type_name(GLenum shader_object_type) -> czstring {
  switch (shader_object_type) {
    case GL_VERTEX_SHADER:
      return "vertex";
      break;

    case GL_TESS_CONTROL_SHADER:
      return "tessellation control";
      break;

    case GL_TESS_EVALUATION_SHADER:
      return "tessellation evaluation";
      break;

    case GL_GEOMETRY_SHADER:
      return "geometry";
      break;

    case GL_FRAGMENT_SHADER:
      return "fragment";
      break;

    case GL_COMPUTE_SHADER:
      return "compute";
      break;

    default:
      return "unknown";
  }
}

template <GLenum shader_object_type>
class shader_object final : public shader_object_handle {
 public:
  using base = shader_object_handle;
  using const_base = const shader_object_handle;

  static constexpr auto type() noexcept { return shader_object_type; }
  static constexpr auto type_name() noexcept {
    return shader_object_type_name(type());
  }

  // operator ref<>() noexcept { return handle; }
  // operator const_ref<shader_object<shader_object_type>>() const noexcept {
  //   return handle;
  // }
  // auto ref() noexcept -> shader_object_ref { return handle; }
  // auto ref() const noexcept -> shader_object_const_ref { return handle; }

  shader_object() {
    handle = glCreateShader(shader_object_type);
    // Not receiving a valid handle is exceptional.
    if (!handle)
      throw runtime_error(format(
          "Failed to receive handle for {} shader object.", type_name()));
  }

  shader_object(auto&&... str) : shader_object{} {
    set_source(forward<decltype(str)>(str)...);
    compile();
  }

  ~shader_object() noexcept { glDeleteShader(handle); }

  // Copying is NOT allowed.
  //
  shader_object(const shader_object&) = delete;
  shader_object& operator=(const shader_object&) = delete;

  // Moving is allowed.
  //
  shader_object(shader_object&& x) : base{x.handle} { x.handle = 0; }
  shader_object& operator=(shader_object&& x) {
    swap(handle, x.handle);
    return *this;
  }
};

// template <GLenum shader_object_type>
// struct ref<shader_object<shader_object_type>>
//     : shader_object<shader_object_type>::base {};

// template <GLenum shader_object_type>
// struct const_ref<shader_object<shader_object_type>>
//     : shader_object<shader_object_type>::const_base {
//   using base = shader_object<shader_object_type>::const_base;
//   // using base::handle;
//   constexpr const_ref(const shader_object<shader_object_type>& shader) noexcept
//       : base{shader.id()} {}
// };

using vertex_shader = shader_object<GL_VERTEX_SHADER>;
using tessellation_control_shader = shader_object<GL_TESS_CONTROL_SHADER>;
using tessellation_evaluation_shader = shader_object<GL_TESS_EVALUATION_SHADER>;
using geometry_shader = shader_object<GL_GEOMETRY_SHADER>;
using fragment_shader = shader_object<GL_FRAGMENT_SHADER>;
using compute_shader = shader_object<GL_COMPUTE_SHADER>;

inline auto vertex_shader_from_file(const filesystem::path& path)
    -> vertex_shader {
  return vertex_shader{string_from_file(path)};
}

inline auto tessellation_control_shader_from_file(const filesystem::path& path)
    -> tessellation_control_shader {
  return tessellation_control_shader{string_from_file(path)};
}

inline auto tessellation_evaluation_shader_from_file(
    const filesystem::path& path) -> tessellation_evaluation_shader {
  return tessellation_evaluation_shader{string_from_file(path)};
}

inline auto geometry_shader_from_file(const filesystem::path& path)
    -> geometry_shader {
  return geometry_shader{string_from_file(path)};
}

inline auto fragment_shader_from_file(const filesystem::path& path)
    -> fragment_shader {
  return fragment_shader{string_from_file(path)};
}

inline auto compute_shader_from_file(const filesystem::path& path)
    -> compute_shader {
  return compute_shader{string_from_file(path)};
}

}  // namespace demo::opengl
