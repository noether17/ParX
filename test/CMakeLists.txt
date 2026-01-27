set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -O3")

add_executable(SingleThreadedDoerTest SingleThreadedDoerTest.cpp)
target_link_libraries(SingleThreadedDoerTest
  GTest::GTest
  GTest::Main
  ParD::ParD
)
add_test(NAME SingleThreadedDoerTest COMMAND SingleThreadedDoerTest)
