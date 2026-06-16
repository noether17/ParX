#pragma once

#include <atomic>
#include <barrier>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "ParX/ParallelExecutor.hpp"
#include "ParX/util/Logging.hpp"

namespace ParX {
/* This class template provides a lock-free queue. It assumes that there are
 * several consumer threads non-destructively reading the front of the queue,
 * but only one consumer thread calling pop() (after all other consumers have
 * called front()) and one producer thread calling push().
 */
inline constexpr auto cache_line_size =
    std::hardware_destructive_interference_size;
template <typename T, std::size_t buffer_size>
class MultiReaderQ {
 public:
  MultiReaderQ()
      : buffer_{static_cast<T*>(std::malloc(buffer_size * sizeof(T)))} {}
  ~MultiReaderQ() { std::free(buffer_); }

  auto pop() {
    auto element = std::move(buffer_[front_index_ % buffer_size]);
    buffer_[front_index_ % buffer_size].~T();
    ++front_index_;
    size_.fetch_sub(1, std::memory_order_relaxed);
    return element;
  }

  bool try_push(T const& t) {
    if (size_.load(std::memory_order_relaxed) >= buffer_size) {
      return false;
    }
    new (&buffer_[back_index_ % buffer_size]) T(t);
    ++back_index_;
    size_.fetch_add(1, std::memory_order_release);
    return true;
  }

  /* Waits for the queue to become empty.
   */
  void await() const {
    auto current_size = std::size_t{};
    while ((current_size = size_.load(std::memory_order_relaxed)) != 0) {
    }
  }

  bool empty() const { return size_.load(std::memory_order_acquire) == 0; }

 private:
  T* const buffer_{};
  alignas(cache_line_size) std::atomic<std::size_t> size_{};
  alignas(cache_line_size) std::size_t front_index_{};
  alignas(cache_line_size) std::size_t back_index_{};
};

struct Task {
  std::function<void(std::size_t, std::size_t)> operation{};
  std::size_t n_items{};
};

template <typename Testable, typename Predicate>
void busy_wait(Testable&& t, Predicate&& pred) {
  for (auto trial = 0; not pred(t); ++trial) {
    __builtin_ia32_pause();
    if (trial == 16) {
      trial = 0;
      std::this_thread::yield();
    }
  }
}

template <typename Testable, typename Predicate>
void busy_wait(Testable&& t, Predicate&& pred)
  requires requires(Testable t_, std::memory_order order) { t_.load(order); }
{
  // Performs busy_wait() using relaxed memory order before final check using
  // acquire memory order.
  do {
    busy_wait([&] { return t.load(std::memory_order_relaxed); },
              [&](auto&& t_lambda) { return pred(t_lambda()); });
  } while (not pred(t.load(std::memory_order_acquire)));
}

class TPE4 {
 public:
  explicit TPE4(std::size_t n_threads) : task_ready_flags_(n_threads) {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      threads_.emplace_back([this, thread_id, n_threads] {
        SCOPED_LOG("thread ", thread_id);
        auto cached_n_items = std::size_t{};
        auto thread_begin_idx = std::size_t{};
        auto thread_end_idx = std::size_t{};
        while (true) {
          if (thread_id == 0) {
            // Wait for all worker threads to finish.
            while (n_running_threads_.load(std::memory_order_acquire) != 0) {
            }

            // Wait for next task.
            while (task_queue_.empty()) {
            }
            n_running_threads_.store(n_threads);

            // Publish active_task_.
            active_task_ = task_queue_.pop();
            for (auto& f : task_ready_flags_) {
              f.store(true, std::memory_order_release);
            }
          } else {
            // Wait for task ready flag.
            busy_wait(task_ready_flags_[thread_id],
                      [](auto flag) { return flag; });
            task_ready_flags_[thread_id].store(false,
                                               std::memory_order_relaxed);
          }

          if (active_task_.operation == nullptr) {
            FUNCTION_LOG("thread ", thread_id, " stopping work.");
            n_running_threads_.fetch_sub(1, std::memory_order_relaxed);
            break;  // stop-work issued.
          }
          if (cached_n_items != active_task_.n_items) {
            cached_n_items = active_task_.n_items;
            auto const items_per_thread =
                (active_task_.n_items + n_threads - 1) / n_threads;
            thread_begin_idx = thread_id * items_per_thread;
            thread_end_idx = std::min((thread_id + 1) * items_per_thread,
                                      active_task_.n_items);
          }
          FUNCTION_LOG("thread ", thread_id, " executing task.");
          active_task_.operation(thread_begin_idx, thread_end_idx);
          n_running_threads_.fetch_sub(1, std::memory_order_release);
        }
      });
    }
  }

  ~TPE4() {
    SCOPED_LOG();
    while (not task_queue_.try_push({}));  // Null task indicates stop-work.
  }

  auto synchronize() const {
    SCOPED_LOG();
    task_queue_.await();
    while (n_running_threads_.load(std::memory_order_acquire) != 0) {
    }
  }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    SCOPED_LOG();
    auto task =
        Task{[... args = std::move(args)](std::size_t thread_begin_idx,
                                          std::size_t thread_end_idx) {
               for (auto i = thread_begin_idx; i < thread_end_idx; ++i) {
                 kernel(i, args...);
               }
             },
             n_items};
    while (not task_queue_.try_push(task)) {
    }
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  static constexpr void transform_reduce_kernel(
      std::size_t thread_id, std::optional<T>* thread_partial_results,
      std::size_t n_items, std::size_t n_items_per_thread,
      TransformArgs... transform_args)
    requires(Reduction<reduce, T> and Transform<transform, T, TransformArgs...>)
  {
    if (auto const initial_index = thread_id * n_items_per_thread;
        initial_index < n_items) {
      auto thread_partial_result = transform(initial_index, transform_args...);
      for (auto i = initial_index + 1;
           i < initial_index + n_items_per_thread and i < n_items; ++i) {
        auto transform_result = transform(i, transform_args...);
        thread_partial_result = reduce(thread_partial_result, transform_result);
      }
      thread_partial_results[thread_id] = thread_partial_result;
    }
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, std::size_t n_items,
                        TransformArgs... transform_args)
    requires(Reduction<reduce, T> and Transform<transform, T, TransformArgs...>)
  {
    SCOPED_LOG();
    auto const n_threads = std::ssize(threads_);
    auto thread_partial_results = std::vector<std::optional<T>>(
        n_threads);  // TODO: Remove this allocation (likely need
                     // n_threads to be a template parameter so
                     // std::array can be used instead).
    auto n_items_per_thread = (n_items + n_threads - 1) / n_threads;
    call_kernel<
        transform_reduce_kernel<T, reduce, transform, TransformArgs...>>(
        n_threads, thread_partial_results.data(), n_items, n_items_per_thread,
        transform_args...);
    synchronize();
    FUNCTION_LOG("synchronized threads.");
    auto result = init_val;
    for (auto partial_result_iter = thread_partial_results.begin();
         partial_result_iter != thread_partial_results.end() and
         (*partial_result_iter).has_value();
         ++partial_result_iter) {
      result = reduce(result, (*partial_result_iter).value());
    }
    return result;
  }

  constexpr auto n_threads() const { return std::ssize(threads_); }

 private:
  alignas(cache_line_size) MultiReaderQ<Task, 16> task_queue_{};
  alignas(cache_line_size) Task active_task_{};
  alignas(cache_line_size) std::atomic_int n_running_threads_{};
  struct alignas(cache_line_size) Flag : std::atomic_bool {};
  alignas(cache_line_size) std::vector<Flag> task_ready_flags_{};
  std::vector<std::jthread> threads_{};
};

using AsyncAtomicQueueThreadPoolExecutor2 = TPE4;

// Template version for testing purposes.
template <std::size_t num_threads>
struct AsyncAtomicQueueThreadPoolTemplateExecutor2
    : AsyncAtomicQueueThreadPoolExecutor2 {
  AsyncAtomicQueueThreadPoolTemplateExecutor2()
      : AsyncAtomicQueueThreadPoolExecutor2(num_threads) {}
};
}  // namespace ParX
