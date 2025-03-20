#pragma once
#include "defaults.hpp"

namespace demo {

struct file_watcher {
  struct entry_info {
    std::filesystem::file_time_type timestamp;
    std::function<void()> callback;
  };

  std::map<std::filesystem::path, entry_info> entries{};

  void _watch(auto&& callback, auto&&... paths) {
    auto f = std::bind<void>(auto(callback), paths...);
    ((entries[paths] = entry_info{last_write_time(paths), f}), ...);
  }

  void watch(auto&& callback, auto&&... paths) {
    _watch(std::forward<decltype(callback)>(callback), canonical(paths)...);
  }

  void process() {
    std::vector<std::function<void()>*> callbacks{};
    for (auto& [path, info] : entries) {
      if (!exists(path)) continue;
      const auto timestamp = last_write_time(path);
      if (info.timestamp == timestamp) continue;
      info.timestamp = timestamp;
      // Do not invoke callback, yet.
      // Callbacks are allowed to call `watch` again.
      // This might lead to pointer and iterator invalidation and could lead to
      // segmentation faults and other memory safety issues during this for loop.
      callbacks.push_back(&info.callback);
    }
    for (auto f : callbacks) std::invoke(*f);
  }
};

}  // namespace demo
