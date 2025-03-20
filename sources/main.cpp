#include "viewer.hpp"

int main(int argc, char* argv[]) {
  demo::viewer viewer{};

  if (argc > 1) viewer.eval_lua_file(argv[1]);

  // if (argc > 1) {
  //   viewer.load_scene_from_file(argv[1]);
  //   viewer.select_maxmin_vids();
  // }
  // if (argc > 2) viewer.load_background_from_file(argv[2]);

  viewer.run();
}
