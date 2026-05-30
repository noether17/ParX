include_guard()

add_custom_target("SingleThreadedExecutor")
set_target_properties("SingleThreadedExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::SingleThreadedExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/SingleThreadedExecutor.hpp"
)
add_custom_target("ThreadPoolExecutor")
set_target_properties("ThreadPoolExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::ThreadPoolExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/ThreadPoolExecutor.hpp"
  PAR_EXEC_CTOR_PARAMS "CPU_THREADS"
)
add_custom_target("ThreadPoolTemplateExecutor")
set_target_properties("ThreadPoolTemplateExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::ThreadPoolTemplateExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/ThreadPoolExecutor.hpp"
  PAR_EXEC_T_PARAMS "CPU_THREADS"
)
add_custom_target("AsyncThreadPoolExecutor")
set_target_properties("AsyncThreadPoolExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::AsyncThreadPoolExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/AsyncThreadPoolExecutor.hpp"
  PAR_EXEC_CTOR_PARAMS "CPU_THREADS"
)
add_custom_target("AsyncThreadPoolTemplateExecutor")
set_target_properties("AsyncThreadPoolTemplateExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::AsyncThreadPoolTemplateExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/AsyncThreadPoolExecutor.hpp"
  PAR_EXEC_T_PARAMS "CPU_THREADS"
)
add_custom_target("CudaExecutor")
set_target_properties("CudaExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::CudaExecutor"
  PAR_EXEC_LANG "CUDA"
  PAR_EXEC_HEADER "ParX/CudaExecutor.cuh"
  PAR_EXEC_T_PARAMS "CUDA_THREADS_PER_BLOCK"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "SingleThreadedExecutor"
  "ThreadPoolExecutor"
  "ThreadPoolTemplateExecutor"
  "AsyncThreadPoolExecutor"
  "AsyncThreadPoolTemplateExecutor"
  "CudaExecutor"
)
