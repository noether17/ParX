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

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    -Xcompiler -Werror \
    -Xcompiler -O3 \
    --expt-relaxed-constexpr"
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

get_property(PAR_EXECS GLOBAL PROPERTY PARALLEL_EXECUTORS)
foreach(PAR_EXEC IN LISTS PAR_EXECS)
  message(STATUS "Reading ${PAR_EXEC} from list PAR_EXECS")
  get_target_property(PAR_EXEC_LANG "${PAR_EXEC}" "LANGUAGE")
  message(STATUS "  ${PAR_EXEC}::LANGUAGE: ${PAR_EXEC_LANG}")
  get_target_property(PAR_EXEC_TMP_PARAMS "${PAR_EXEC}" "TEMPLATE_PARAMS")
  message(STATUS "  ${PAR_EXEC}::TEMPLATE_PARAMS: ${PAR_EXEC_TMP_PARAMS}")
  get_target_property(PAR_EXEC_HEADER "${PAR_EXEC}" "HEADER_FILE")
  message(STATUS "  ${PAR_EXEC}::HEADER_FILE: ${PAR_EXEC_HEADER}")
endforeach()

function(generate_par_exec_test
    PAR_EXEC
    PAR_EXEC_HEADER
    TEMPLATE_ARGS
    CTOR_ARGS
    LANGUAGE
  )
  if(TEMPLATE_ARGS)
    set(PAR_EXEC_TYPE "${PAR_EXEC}<${TEMPLATE_ARGS}>")
  else()
    set(PAR_EXEC_TYPE "${PAR_EXEC}")
  endif()
  string(REGEX REPLACE "[<,]" "_" PAR_EXEC_TYPE_TOKEN "${PAR_EXEC_TYPE}")
  string(REGEX REPLACE "::" "_" PAR_EXEC_TYPE_TOKEN "${PAR_EXEC_TYPE_TOKEN}")
  string(REGEX REPLACE "[> ]" "" PAR_EXEC_TYPE_TOKEN "${PAR_EXEC_TYPE_TOKEN}")

  if(LANGUAGE MATCHES "CXX")
    set(TEST_FILE_EXT "cpp")
  elseif(LANGUAGE MATCHES "CUDA")
    set(TEST_FILE_EXT "cu")
  else()
    message(STATUS "LANGUAGE not recognized")
  endif()

  set(PAR_EXEC_SPEC "${PAR_EXEC_TYPE_TOKEN}_${CTOR_ARGS}")
  string(REGEX REPLACE ", *" "_" PAR_EXEC_SPEC "${PAR_EXEC_SPEC}")
  string(REGEX REPLACE "_$" "" PAR_EXEC_SPEC "${PAR_EXEC_SPEC}")
  set(TEST_NAME "${PAR_EXEC_SPEC}_Test")
  set(TEST_FILE_NAME "${TEST_NAME}.${TEST_FILE_EXT}")
  set(TEST_FILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/generated/${TEST_FILE_NAME}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ParallelExecutorTest.cpp.in"
    "${TEST_FILE_PATH}"
  )
  message(STATUS "Configured ${TEST_FILE_PATH}")

  add_executable("${TEST_NAME}" "${TEST_FILE_PATH}")
  target_link_libraries("${TEST_NAME}"
    GTest::GTest
    GTest::Main
    ParX::ParX
  )
  add_test(NAME "${TEST_NAME}" COMMAND "${TEST_NAME}")
endfunction()

#generate_par_exec_test(
#  "ParX::SingleThreadedExecutor"
#  "ParX/SingleThreadedExecutor.hpp"
#  ""
#  ""
#  "CXX"
#)
#generate_par_exec_test(
#  "ParX::ThreadPoolExecutor"
#  "ParX/ThreadPoolExecutor.hpp"
#  ""
#  "32"
#  "CXX"
#)
#generate_par_exec_test(
#  "ParX::ThreadPoolTemplateExecutor"
#  "ParX/ThreadPoolExecutor.hpp"
#  "32"
#  ""
#  "CXX"
#)
#generate_par_exec_test(
#  "ParX::CudaExecutor"
#  "ParX/CudaExecutor.cuh"
#  "32"
#  ""
#  "CUDA"
#)

set(CPU_THREADS 1 2 4 8 16 32)
set(CUDA_THREADS_PER_BLOCK 32 64 128 256 512 1024)

function(generate_par_exec_test_files
    PAR_EXEC
    TEMPLATE_PARAM_GROUPS
    PARTIAL_TEMPLATE_ARGS
    CTOR_PARAM_GROUPS
    PARTIAL_CTOR_ARGS
  )
  #message(STATUS "generate_par_exec_test_files(${PAR_EXEC} ${TEMPLATE_PARAM_GROUPS} ${PARTIAL_TEMPLATE_ARGS} ${CTOR_PARAM_GROUPS} ${PARTIAL_CTOR_ARGS})")
  if(TEMPLATE_PARAM_GROUPS)
    list(POP_FRONT TEMPLATE_PARAM_GROUPS CURRENT_TEMPLATE_PARAM)
    #list(GET TEMPLATE_PARAM_GROUPS 0 CURRENT_TEMPLATE_PARAM)
    #message(STATUS "CURRENT_TEMPLATE_PARAM: ${CURRENT_TEMPLATE_PARAM}")
    foreach(TEMPLATE_ARG IN LISTS "${CURRENT_TEMPLATE_PARAM}")
      #message(STATUS "TEMPLATE_ARG: ${TEMPLATE_ARG}")
      if(PARTIAL_TEMPLATE_ARGS)
        set(CURRENT_TEMPLATE_ARGS "${PARTIAL_TEMPLATE_ARGS}, ${TEMPLATE_ARG}")
      else()
        set(CURRENT_TEMPLATE_ARGS "${TEMPLATE_ARG}")
      endif()
      generate_par_exec_test_files("${PAR_EXEC}"
        "${TEMPLATE_PARAM_GROUPS}"
        "${CURRENT_TEMPLATE_ARGS}"
        "${CTOR_PARAM_GROUPS}"
        "${PARTIAL_CTOR_ARGS}"
      )
    endforeach()
  else()
    if(CTOR_PARAM_GROUPS)
      list(POP_FRONT CTOR_PARAM_GROUPS CURRENT_CTOR_PARAM)
      #message(STATUS "CURRENT_CTOR_PARAM: ${CURRENT_CTOR_PARAM}")
      foreach(CTOR_ARG IN LISTS "${CURRENT_CTOR_PARAM}")
        #message(STATUS "CTOR_ARG: ${CTOR_ARG}")
        if(PARTIAL_CTOR_ARGS)
          set(CURRENT_CTOR_ARGS "${PARTIAL_CTOR_ARGS}, ${CTOR_ARG}")
        else()
          set(CURRENT_CTOR_ARGS "${CTOR_ARG}")
        endif()
        generate_par_exec_test_files("${PAR_EXEC}"
          "${TEMPLATE_PARAM_GROUPS}"
          "${PARTIAL_TEMPLATE_ARGS}"
          "${CTOR_PARAM_GROUPS}"
          "${CURRENT_CTOR_ARGS}"
        )
      endforeach()
    else()
      if(PARTIAL_TEMPLATE_ARGS)
        message(STATUS "${PAR_EXEC}<${PARTIAL_TEMPLATE_ARGS}>(${PARTIAL_CTOR_ARGS})")
      else()
        message(STATUS "${PAR_EXEC}(${PARTIAL_CTOR_ARGS})")
      endif()
      generate_par_exec_test("${PAR_EXEC}"
        "${PAR_EXEC_HEADER}"
        "${PARTIAL_TEMPLATE_ARGS}"
        "${PARTIAL_CTOR_ARGS}"
        "${PAR_EXEC_LANG}"
      )
    endif()
  endif()
endfunction()

foreach(PAR_EXEC IN LISTS PAR_EXECS)
  message(STATUS "PAR_EXEC: ${PAR_EXEC}")
  get_target_property(PAR_EXEC_NAME "${PAR_EXEC}" "PAR_EXEC_NAME")
  message(STATUS "  PAR_EXEC_NAME: ${PAR_EXEC_NAME}")
  get_target_property(PAR_EXEC_LANG "${PAR_EXEC}" "PAR_EXEC_LANG")
  message(STATUS "  PAR_EXEC_LANG: ${PAR_EXEC_LANG}")
  get_target_property(PAR_EXEC_HEADER "${PAR_EXEC}" "PAR_EXEC_HEADER")
  message(STATUS "  PAR_EXEC_HEADER: ${PAR_EXEC_HEADER}")
  get_target_property(PAR_EXEC_T_PARAMS "${PAR_EXEC}" "PAR_EXEC_T_PARAMS")
  message(STATUS "  PAR_EXEC_T_PARAMS: ${PAR_EXEC_T_PARAMS}")
  if(NOT PAR_EXEC_T_PARAMS)
    message(STATUS "    No template parameters.")
  endif()
  get_target_property(PAR_EXEC_CTOR_PARAMS "${PAR_EXEC}" "PAR_EXEC_CTOR_PARAMS")
  message(STATUS "  PAR_EXEC_CTOR_PARAMS: ${PAR_EXEC_CTOR_PARAMS}")
  if(NOT PAR_EXEC_CTOR_PARAMS)
    message(STATUS "    No constructor parameters.")
  endif()
  generate_par_exec_test_files("${PAR_EXEC_NAME}"
    "${PAR_EXEC_T_PARAMS}"
    ""
    "${PAR_EXEC_CTOR_PARAMS}"
    ""
  )
endforeach()
