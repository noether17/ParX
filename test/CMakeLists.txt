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

add_executable(ThreadPoolDoerTest ThreadPoolDoerTest.cpp)
target_link_libraries(ThreadPoolDoerTest
  GTest::GTest
  GTest::Main
  ParD::ParD
)
add_test(NAME ThreadPoolDoerTest COMMAND ThreadPoolDoerTest)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA is available; including CUDA tests.")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
  set(CMAKE_CUDA_ARCHITECTURES native)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    -Xcompiler -Werror \
    -Xcompiler -O3 \
    --expt-relaxed-constexpr"
  )

  add_executable(CudaDoerTest CudaDoerTest.cu)
  target_link_libraries(CudaDoerTest
    GTest::GTest
    GTest::Main
    ParD::ParD
  )
  add_test(NAME CudaDoerTest COMMAND CudaDoerTest)
else()
  message(STATUS "CUDA is unavailable; skipping CUDA tests.")
endif()
