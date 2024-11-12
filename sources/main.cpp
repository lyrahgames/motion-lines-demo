#include "viewer.hpp"

int main(int argc, char* argv[]) {
  demo::viewer viewer{};

  if (argc > 1) viewer.load_scene_from_file(argv[1]);

  viewer.run();
}
