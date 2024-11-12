#include "log.hpp"

namespace demo::log {

static void log(auto&& prefix, auto&& str, source_location location) {
  fmt::print("{}{}\n\n{}\n\n", forward<decltype(prefix)>(prefix),
             string_from(location), forward<decltype(str)>(str));
}

void text(std::string_view str) {
  std::println("{}", str);
}

auto string_from(source_location location) -> string {
  return fmt::format(
      fg(fmt::color::gray), "{}\n{}:",
      fmt::format("{}:{}:{}: ",  //
                  relative(filesystem::path(location.file_name())).string(),
                  location.line(), location.column()),
      location.function_name());
}

void debug(const string& str, source_location location) {
  log(fmt::format(fg(fmt::color::gray), "DEBUG:   \n"), str, location);
}

void info(const string& str, source_location location) {
  log(fmt::format(fg(fmt::color::green), "INFO:    \n"), str, location);
}

void warn(const string& str, source_location location) {
  log(fmt::format(fg(fmt::color::orange), "WARNING: \n"), str, location);
}

void error(const string& str, source_location location) {
  log(fmt::format(fg(fmt::color::red), "ERROR:   \n"), str, location);
}

}  // namespace demo::log
