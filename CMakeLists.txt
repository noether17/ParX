cmake_minimum_required(VERSION 4.0.0)
project(ParX VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

add_library("${PROJECT_NAME}" INTERFACE)
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")
target_include_directories("${PROJECT_NAME}" INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features("${PROJECT_NAME}" INTERFACE cxx_std_20)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" TYPE INCLUDE)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${LIB_INSTALL_DIR}"
  PATH_VARS LIB_INSTALL_DIR
)
install(
  TARGETS "${PROJECT_NAME}"
  EXPORT "${PROJECT_NAME}Targets"
)
install(
  EXPORT "${PROJECT_NAME}Targets"
  NAMESPACE "${PROJECT_NAME}::"
  DESTINATION "${LIB_INSTALL_DIR}"
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION
    "${LIB_INSTALL_DIR}"
)

add_custom_target("SingleThreadedExecutor")
set_target_properties("SingleThreadedExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::SingleThreadedExecutor"
  LANGUAGE "CXX"
  HEADER_FILE "ParX/SingleThreadedExecutor.hpp"
)
add_custom_target("ThreadPoolExecutor")
set_target_properties("ThreadPoolExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::ThreadPoolExecutor"
  LANGUAGE "CXX"
  CTOR_PARAMS "CPU_THREADS"
  HEADER_FILE "ParX/ThreadPoolExecutor.hpp"
)
add_custom_target("ThreadPoolTemplateExecutor")
set_target_properties("ThreadPoolTemplateExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::ThreadPoolTemplateExecutor"
  LANGUAGE "CXX"
  TEMPLATE_PARAMS "CPU_THREADS"
  HEADER_FILE "ParX/ThreadPoolExecutor.hpp"
)
add_custom_target("CudaExecutor")
set_target_properties("CudaExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::CudaExecutor"
  LANGUAGE "CUDA"
  TEMPLATE_PARAMS "CUDA_THREADS_PER_BLOCK"
  HEADER_FILE "ParX/CudaExecutor.cuh"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "SingleThreadedExecutor"
  "ThreadPoolExecutor"
  "ThreadPoolTemplateExecutor"
  "CudaExecutor"
)
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

enable_testing()
find_package(GTest 1.14.0 REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_subdirectory(test)
