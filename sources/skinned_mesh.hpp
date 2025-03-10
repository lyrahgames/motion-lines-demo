#pragma once
#include "camera.hpp"
#include "scene.hpp"

namespace demo {

struct triangle_mesh {
  using vertex = scene::mesh::vertex;
  using face = scene::mesh::face;
  struct group_offset {
    uint32 vertex{};
    uint32 face{};
  };

  std::vector<vertex> vertices{};
  std::vector<face> faces{};
  std::vector<group_offset> group_offsets{};
};

auto triangle_mesh_from(const scene& in) -> triangle_mesh;

struct skeleton {
  auto size() const noexcept -> uint32 { return parents.size(); }
  std::vector<uint32> parents{};
  std::vector<glm::mat4> transforms{};
  std::vector<glm::mat4> offsets{};
};

auto skeleton_from(const scene& in) -> skeleton;

// compressed by vertices
struct weight_matrix {
  struct entry {
    uint32 index{};
    float32 weight{};
  };
  std::vector<uint32> offsets{};
  std::vector<entry> entries{};
};

struct skinned_mesh : triangle_mesh {
  struct animation {
    struct channel {
      template <typename type>
      struct key {
        float32 time{};
        type data{};
      };
      using position_key = key<glm::vec3>;
      using rotation_key = key<glm::quat>;
      using scaling_key = key<glm::vec3>;

      uint32 index;
      std::vector<position_key> positions{};
      std::vector<rotation_key> rotations{};
      std::vector<scaling_key> scalings{};

      auto position(float32 time) const -> glm::mat4;
      auto rotation(float32 time) const -> glm::mat4;
      auto scaling(float32 time) const -> glm::mat4;
      auto transform(float32 time) const -> glm::mat4;
    };

    float32 duration{};
    float32 ticks{};
    std::vector<channel> channels{};
  };

  skeleton bones{};
  weight_matrix weights{};
  std::vector<animation> animations{};
};

auto skinned_mesh_from(const scene& in) -> skinned_mesh;

void load_global_transforms(const skinned_mesh& mesh, auto&& out) {
  auto& bones = mesh.bones;
  assert(std::ranges::size(out) == bones.size());
  std::ranges::copy(bones.transforms, std::ranges::begin(out));
  for (size_t bid = 1; bid < bones.size(); ++bid)
    out[bid] = out[bones.parents[bid]] * out[bid];
  for (size_t bid = 0; bid < bones.size(); ++bid)
    out[bid] *= bones.offsets[bid];
}

inline auto global_transforms(const skinned_mesh& mesh)
    -> std::vector<glm::mat4> {
  std::vector<glm::mat4> result;
  result.resize(mesh.bones.size());
  load_global_transforms(mesh, result);
  return result;
}

void load_animation_transforms(const skinned_mesh& mesh,
                               size_t aid,
                               float32 time,
                               auto&& out) {
  assert(aid < mesh.animations.size());
  auto& animation = mesh.animations[aid];
  auto& bones = mesh.bones;
  assert(std::ranges::size(out) == bones.size());

  std::ranges::fill(out, glm::mat4{1.0f});
  for (size_t i = 0; i < animation.channels.size(); ++i) {
    auto& channel = animation.channels[i];
    out[channel.index] = channel.transform(time * animation.ticks);
  }
  for (size_t bid = 1; bid < bones.size(); ++bid)
    out[bid] = out[bones.parents[bid]] * out[bid];
  for (size_t bid = 0; bid < bones.size(); ++bid)
    out[bid] *= bones.offsets[bid];
}

inline auto animation_transforms(const skinned_mesh& mesh,
                                 size_t aid,
                                 float32 time) -> std::vector<glm::mat4> {
  std::vector<glm::mat4> result;
  result.resize(mesh.bones.size());
  load_animation_transforms(mesh, aid, time, result);
  return result;
}

auto weighted_transform(const skinned_mesh& mesh, auto&& transforms, uint32 vid)
    -> glm::mat4 {
  glm::mat4 result(0.0f);
  for (auto i = mesh.weights.offsets[vid];  //
       i < mesh.weights.offsets[vid + 1]; ++i) {
    const auto [index, weight] = mesh.weights.entries[i];
    result += weight * transforms[index];
  }
  return result;
}

struct sampled_animation {
  struct vertex : triangle_mesh::vertex {
    float32 time;
    float32 length;
  };

  std::vector<vertex> samples{};
  size_t sample_count{};
  float32 time{};
  float32 max_length{};
};

inline auto sampled_animation_from(const skinned_mesh& mesh,
                                   size_t aid,
                                   size_t fps = 60) -> sampled_animation {
  sampled_animation result{};

  result.time = mesh.animations[aid].duration / mesh.animations[aid].ticks;
  result.sample_count = static_cast<size_t>(std::floor(fps * result.time));
  result.samples.resize(mesh.vertices.size() * result.sample_count);

  const auto time_step = result.time / result.sample_count;

  auto transforms = animation_transforms(mesh, aid, 0.0f);
  for (size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
    const auto x = weighted_transform(mesh, transforms, vid);
    const auto index = vid * result.sample_count;
    auto& v = result.samples[index];
    v.position = glm::vec3(x * glm::vec4(mesh.vertices[vid].position, 1.0f));
    v.normal = glm::vec3(transpose(inverse(x)) *
                         glm::vec4(mesh.vertices[vid].normal, 0.0f));
    v.time = 0.0f;
    v.length = 0.0f;
  }

  for (size_t s = 1; s < result.sample_count; ++s) {
    const auto time = s * time_step;
    load_animation_transforms(mesh, aid, time, transforms);

    for (size_t vid = 0; vid < mesh.vertices.size(); ++vid) {
      const auto x = weighted_transform(mesh, transforms, vid);
      const auto index = vid * result.sample_count + s;
      auto& v = result.samples[index];
      auto& p = result.samples[index - 1];
      v.position = glm::vec3(x * glm::vec4(mesh.vertices[vid].position, 1.0f));
      v.normal = glm::vec3(transpose(inverse(x)) *
                           glm::vec4(mesh.vertices[vid].normal, 0.0f));
      v.time = time;
      v.length = p.length + glm::distance(p.position, v.position);
    }
  }

  for (size_t vid = 0; vid < mesh.vertices.size(); ++vid)
    result.max_length =
        std::max(result.max_length,
                 result.samples[(vid + 1) * result.sample_count - 1].length);

  return result;
}

// template <typename type>
// struct list_bundle {
//   using value_type = type;
//   using size_type = uint32;
//   std::vector<value_type> entries{};
//   std::vector<size_type> offsets{0};
// };

struct motion_line_bundle {
  struct vertex {
    vec4 position;
    vec2 pixels;
    vec2 out;
    float32 time;
    float32 arc;
    float32 parc;
    float32 depth;
  };
  std::vector<vertex> vertices{};
  std::vector<uint32> offsets{0};
  std::vector<float32> arcs{};
};

inline auto uniform_motion_line_bundle(const skinned_mesh& mesh,
                                       const auto& seeds,
                                       size_t aid,
                                       size_t fps = 60) -> motion_line_bundle {
  motion_line_bundle bundle{};

  std::vector<glm::mat4> transforms;
  transforms.resize(mesh.bones.size());

  const auto duration =
      mesh.animations[aid].duration / mesh.animations[aid].ticks;

  for (auto vid : seeds) {
    const auto sample_count = static_cast<size_t>(std::floor(fps * duration));
    const auto dt = duration / (sample_count - 1);

    const auto first = bundle.offsets.back();

    for (size_t i = 0; i < sample_count; ++i) {
      const auto t = i * dt;

      load_animation_transforms(mesh, aid, t, transforms);
      const auto m = weighted_transform(mesh, transforms, vid);

      motion_line_bundle::vertex v{};
      v.position = m * vec4(mesh.vertices[vid].position, 1.0f);
      v.time = t;
      bundle.vertices.push_back(v);
    }

    bundle.offsets.push_back(bundle.vertices.size());
    const auto last = bundle.offsets.back();

    bundle.vertices[first].arc = 0.0f;
    for (auto k = first; k < last - 1; ++k)
      bundle.vertices[k + 1].arc =
          bundle.vertices[k].arc +
          glm::distance(bundle.vertices[k].position,
                        bundle.vertices[k + 1].position);
  }

  bundle.arcs.resize(bundle.offsets.size() - 1);

  return bundle;
}

inline auto update_strokes(motion_line_bundle& bundle,
                           float32 time,
                           const struct camera& camera) {
  const auto m = camera.projection_matrix() * camera.view_matrix();
  const auto p = camera.viewport_matrix();
  for (auto& v : bundle.vertices) {
    auto x = m * v.position;
    x /= x.w;
    x = p * x;
    v.pixels = vec2(x);
    v.depth = x.z;
  }

  for (size_t i = 0; i < bundle.offsets.size() - 1; ++i) {
    const auto first = bundle.offsets[i];
    const auto last = bundle.offsets[i + 1];

    {
      const auto x = bundle.vertices[first].pixels;
      const auto q = bundle.vertices[first + 1].pixels;
      const auto t = normalize(q - x);
      bundle.vertices[first].out = vec2(-t.y, t.x);
    }
    for (auto k = first + 1; k < last - 1; ++k) {
      const auto p = bundle.vertices[k - 1].pixels;
      const auto x = bundle.vertices[k].pixels;
      const auto q = bundle.vertices[k + 1].pixels;
      const auto xp = normalize(p - x);
      const auto xq = normalize(q - x);
      const auto t = normalize(xq - xp);
      bundle.vertices[k].out = vec2(-t.y, t.x);
    }
    {
      const auto p = bundle.vertices[last - 2].pixels;
      const auto x = bundle.vertices[last - 1].pixels;
      const auto t = -normalize(p - x);
      bundle.vertices[last - 1].out = vec2(-t.y, t.x);
    }
  }

  for (size_t sid = 0; sid < bundle.offsets.size() - 1; ++sid) {
    auto vid = bundle.offsets[sid];
    if (time < bundle.vertices[vid].time) {
      bundle.arcs[sid] = bundle.vertices[vid].arc;
      continue;
    }
    ++vid;
    for (; vid < bundle.offsets[sid + 1]; ++vid) {
      if (time < bundle.vertices[vid].time) {
        const auto t1 = bundle.vertices[vid - 1].time;
        const auto t2 = bundle.vertices[vid].time;
        const auto arc1 = bundle.vertices[vid - 1].arc;
        const auto arc2 = bundle.vertices[vid].arc;
        const auto t = time;
        bundle.arcs[sid] = ((t2 - t) * arc1 + (t - t1) * arc2) / (t2 - t1);
        break;
      }
    }
    if (time >= bundle.vertices[vid - 1].time)
      bundle.arcs[sid] = bundle.vertices[vid - 1].arc;
  }
}

struct line_map_entry {
  uint32 vertex;
  uint32 stroke;
};

inline auto segments(const motion_line_bundle& bundle)
    -> std::vector<line_map_entry> {
  std::vector<line_map_entry> result{};
  for (size_t i = 0; i < bundle.offsets.size() - 1; ++i)
    for (auto j = bundle.offsets[i]; j < bundle.offsets[i + 1] - 1; ++j)
      result.emplace_back(j, i);
  return result;
}

}  // namespace demo
