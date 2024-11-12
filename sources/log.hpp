#pragma once
#include "defaults.hpp"

namespace demo::log {

auto string_from(source_location location) -> string;

void text(std::string_view str);
void debug(const string& str, source_location = source_location::current());
void info(const string& str, source_location = source_location::current());
void warn(const string& str, source_location = source_location::current());
void error(const string& str, source_location = source_location::current());

}  // namespace demo::log
