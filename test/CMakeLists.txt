set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-interference-size -Werror -O3")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA is available; including CUDA tests.")
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES native)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    -Xcompiler -Werror \
    -Xcompiler -O3 \
    --expt-relaxed-constexpr"
  )
else()
  message(STATUS "CUDA is unavailable; skipping CUDA tests.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/GenerateTests.cmake")

set(CPU_THREADS 1 2 4 8 16 32)
set(QUEUE_SIZE 16)
set(CUDA_THREADS_PER_BLOCK 32 64 128 256 512 1024)

get_property(PAR_EXECS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PAR_EXEC IN LISTS PAR_EXECS)
  get_target_property(PAR_EXEC_NAME "${PAR_EXEC}" "PAR_EXEC_NAME")
  get_target_property(PAR_EXEC_LANG "${PAR_EXEC}" "PAR_EXEC_LANG")
  get_target_property(PAR_EXEC_HEADER "${PAR_EXEC}" "PAR_EXEC_HEADER")
  get_target_property(PAR_EXEC_T_PARAMS "${PAR_EXEC}" "PAR_EXEC_T_PARAMS")
  get_target_property(PAR_EXEC_CTOR_PARAMS "${PAR_EXEC}" "PAR_EXEC_CTOR_PARAMS")
  get_target_property(PAR_EXEC_IS_ASYNC "${PAR_EXEC}" "PAR_EXEC_IS_ASYNC")
  if((PAR_EXEC_LANG MATCHES "CXX") OR
     (PAR_EXEC_LANG MATCHES "CUDA" AND CMAKE_CUDA_COMPILER))
    generate_par_exec_test_files("${PAR_EXEC_NAME}"
      "${PAR_EXEC_T_PARAMS}"
      ""
      "${PAR_EXEC_CTOR_PARAMS}"
      ""
      "ParallelExecutorTest.cpp.in"
      ""
    )
    if(PAR_EXEC_IS_ASYNC)
      generate_par_exec_test_files("${PAR_EXEC_NAME}"
        "${PAR_EXEC_T_PARAMS}"
        ""
        "${PAR_EXEC_CTOR_PARAMS}"
        ""
        "AsyncParallelExecutorTest.cpp.in"
        "Async"
      )
    endif()
  endif()
endforeach()

add_executable(CudaArrayTest CudaArrayTest.cu)
target_link_libraries(CudaArrayTest
  GTest::GTest
  GTest::Main
  ParX::ParX
)
add_test(NAME CudaArrayTest COMMAND CudaArrayTest)
