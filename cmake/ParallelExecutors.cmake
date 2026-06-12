include_guard()

add_custom_target("SingleThreadedExecutor")
set_target_properties("SingleThreadedExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::SingleThreadedExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/SingleThreadedExecutor.hpp"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "SingleThreadedExecutor"
)

add_custom_target("ThreadPoolExecutor")
set_target_properties("ThreadPoolExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::ThreadPoolExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/ThreadPoolExecutor.hpp"
  PAR_EXEC_CTOR_PARAMS "CPU_THREADS"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "ThreadPoolExecutor"
)

add_custom_target("ThreadPoolTemplateExecutor")
set_target_properties("ThreadPoolTemplateExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::ThreadPoolTemplateExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/ThreadPoolExecutor.hpp"
  PAR_EXEC_T_PARAMS "CPU_THREADS"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "ThreadPoolTemplateExecutor"
)

add_custom_target("AsyncQueueThreadPoolExecutor")
set_target_properties("AsyncQueueThreadPoolExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::AsyncQueueThreadPoolExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/AsyncQueueThreadPoolExecutor.hpp"
  PAR_EXEC_CTOR_PARAMS "CPU_THREADS"
  PAR_EXEC_IS_ASYNC "YES"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "AsyncQueueThreadPoolExecutor"
)

add_custom_target("AsyncAtomicQueueThreadPoolExecutor")
set_target_properties("AsyncAtomicQueueThreadPoolExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::AsyncAtomicQueueThreadPoolExecutor"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/AsyncAtomicQueueThreadPoolExecutor.hpp"
  PAR_EXEC_CTOR_PARAMS "CPU_THREADS"
  PAR_EXEC_IS_ASYNC "YES"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "AsyncAtomicQueueThreadPoolExecutor"
)

add_custom_target("AsyncAtomicQueueThreadPoolExecutor2")
set_target_properties("AsyncAtomicQueueThreadPoolExecutor2" PROPERTIES
  PAR_EXEC_NAME "ParX::AsyncAtomicQueueThreadPoolExecutor2"
  PAR_EXEC_LANG "CXX"
  PAR_EXEC_HEADER "ParX/AsyncAtomicQueueThreadPoolExecutor2.hpp"
  PAR_EXEC_CTOR_PARAMS "CPU_THREADS"
  PAR_EXEC_IS_ASYNC "YES"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "AsyncAtomicQueueThreadPoolExecutor2"
)

add_custom_target("CudaExecutor")
set_target_properties("CudaExecutor" PROPERTIES
  PAR_EXEC_NAME "ParX::CudaExecutor"
  PAR_EXEC_LANG "CUDA"
  PAR_EXEC_HEADER "ParX/CudaExecutor.cuh"
  PAR_EXEC_T_PARAMS "CUDA_THREADS_PER_BLOCK"
  PAR_EXEC_IS_ASYNC "YES"
)
set_property(GLOBAL APPEND PROPERTY PARALLEL_EXECUTORS
  "CudaExecutor"
)
