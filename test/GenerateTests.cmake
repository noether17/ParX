include_guard()

function(generate_par_exec_test
    PAR_EXEC
    PAR_EXEC_HEADER
    TEMPLATE_ARGS
    CTOR_ARGS
    PAR_EXEC_LANG
  )
  if(TEMPLATE_ARGS)
    set(PAR_EXEC_TYPE "${PAR_EXEC}<${TEMPLATE_ARGS}>")
  else()
    set(PAR_EXEC_TYPE "${PAR_EXEC}")
  endif()
  string(REGEX REPLACE "[<,]" "_" PAR_EXEC_TYPE_TOKEN "${PAR_EXEC_TYPE}")
  string(REGEX REPLACE "::" "_" PAR_EXEC_TYPE_TOKEN "${PAR_EXEC_TYPE_TOKEN}")
  string(REGEX REPLACE "[> ]" "" PAR_EXEC_TYPE_TOKEN "${PAR_EXEC_TYPE_TOKEN}")

  if(PAR_EXEC_LANG MATCHES "CXX")
    set(TEST_FILE_EXT "cpp")
  elseif(PAR_EXEC_LANG MATCHES "CUDA")
    set(TEST_FILE_EXT "cu")
  else()
    message(STATUS "PAR_EXEC_LANG not recognized")
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

function(generate_par_exec_test_files
    PAR_EXEC
    TEMPLATE_PARAM_GROUPS
    PARTIAL_TEMPLATE_ARGS
    CTOR_PARAM_GROUPS
    PARTIAL_CTOR_ARGS
  )
  if(TEMPLATE_PARAM_GROUPS)
    list(POP_FRONT TEMPLATE_PARAM_GROUPS CURRENT_TEMPLATE_PARAM)
    foreach(TEMPLATE_ARG IN LISTS "${CURRENT_TEMPLATE_PARAM}")
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
      foreach(CTOR_ARG IN LISTS "${CURRENT_CTOR_PARAM}")
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
      generate_par_exec_test("${PAR_EXEC}"
        "${PAR_EXEC_HEADER}"
        "${PARTIAL_TEMPLATE_ARGS}"
        "${PARTIAL_CTOR_ARGS}"
        "${PAR_EXEC_LANG}"
      )
    endif()
  endif()
endfunction()
