#pragma once

#define LOGGING_ON
#ifdef LOGGING_ON

#include <iostream>
#include <source_location>
#include <syncstream>

template <typename... Args>
void LOG(Args&&... args) {
  auto sync_cout = std::osyncstream{std::cout};
  (sync_cout << ... << args) << '\n';
}

template <typename... Args>
class ScopedLogger {
 public:
  ScopedLogger(Args&&... args, std::source_location const& loc =
                                   std::source_location::current())
      : loc_{loc} {
    (message_ << ... << args) << "";
    LOG(loc_.function_name(), '(', loc_.file_name(), ':', loc_.line(), ") ",
        message_.str());
  }

  ~ScopedLogger() {
    LOG('~', loc_.function_name(), '(', loc_.file_name(), ':', loc_.line(),
        ") ", message_.str());
  }

 private:
  std::source_location loc_{};
  std::stringstream message_{};
};

template <typename... Args>
class FUNCTION_LOG {
 public:
  FUNCTION_LOG(Args&&... args, std::source_location const& loc =
                                   std::source_location::current()) {
    LOG(loc.function_name(), '(', loc.file_name(), ':', loc.line(),
        "): ", args...);
  }
};

template <typename... Args>
FUNCTION_LOG(Args&&...) -> FUNCTION_LOG<Args...>;

#else
#define FUNCTION_LOG(...)
#endif
