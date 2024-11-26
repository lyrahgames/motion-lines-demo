#include "viewer.hpp"

int main(int argc, char* argv[]) {
  demo::viewer viewer{};

  if (argc > 1) viewer.load_scene_from_file(argv[1]);
  if (argc > 2) viewer.load_vids_from_file(argv[2]);

  viewer.run();
}
