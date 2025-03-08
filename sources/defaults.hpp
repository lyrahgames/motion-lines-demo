#pragma once
//
#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
// #include <fmt/std.h>
//
#include <ensketch/xstd/xstd.hpp>
//
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>
//
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
//
#include <glm/ext.hpp>
//
#include <glm/gtx/norm.hpp>

namespace demo {

using namespace std;
using namespace gl;

using namespace ensketch;
using xstd::czstring;
using xstd::float32;
using xstd::float64;
using xstd::uint32;
//
using xstd::infinity;
using xstd::pi;

using glm::ivec2;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

// Attempt to quit the application in a thread-safe manner.
//
void quit() noexcept;

// Receive an `std::stop_token` from the application's global stop source.
//
auto stop_token() noexcept -> std::stop_token;

// Check whether the application has been requested to stop.
// Internally, this will receive the stop state of the global stop source.
//
bool done() noexcept;

}  // namespace demo
