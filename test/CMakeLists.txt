set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -O3")

add_executable(SingleThreadedExecutorTest SingleThreadedExecutorTest.cpp)
target_link_libraries(SingleThreadedExecutorTest
  GTest::GTest
  GTest::Main
  ParX::ParX
)
add_test(NAME SingleThreadedExecutorTest COMMAND SingleThreadedExecutorTest)

add_executable(ThreadPoolExecutorTest ThreadPoolExecutorTest.cpp)
target_link_libraries(ThreadPoolExecutorTest
  GTest::GTest
  GTest::Main
  ParX::ParX
)
add_test(NAME ThreadPoolExecutorTest COMMAND ThreadPoolExecutorTest)

add_executable(ThreadPoolTemplateExecutorTest ThreadPoolTemplateExecutorTest.cpp)
target_link_libraries(ThreadPoolTemplateExecutorTest
  GTest::GTest
  GTest::Main
  ParX::ParX
)
add_test(NAME ThreadPoolTemplateExecutorTest COMMAND ThreadPoolTemplateExecutorTest)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA is available; including CUDA tests.")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
  set(CMAKE_CUDA_ARCHITECTURES native)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    -Xcompiler -Werror \
    -Xcompiler -O3 \
    -Xcompiler -fconcepts-diagnostics-depth=2 \
    --expt-relaxed-constexpr \
    --extended-lambda \
    -v"
  )

  add_executable(CudaExecutorTest CudaExecutorTest.cu)
  target_link_libraries(CudaExecutorTest
    GTest::GTest
    GTest::Main
    ParX::ParX
  )
  add_test(NAME CudaExecutorTest COMMAND CudaExecutorTest)
else()
  message(STATUS "CUDA is unavailable; skipping CUDA tests.")
endif()
