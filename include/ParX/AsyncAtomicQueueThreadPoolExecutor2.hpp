#pragma once

#include <atomic>
#include <barrier>
#include <cstddef>
#include <functional>
#include <iterator>
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
template <typename T, std::size_t buffer_size>
class MultiReaderQ {
 public:
  /* Non-destructively reads the front element of the queue. Blocks if the queue
   * is empty.
   */
  auto front() const {
    size_.wait(0, std::memory_order_acquire);
    return buffer_[front_index_ % buffer_size];
  }

  /* Waits for the queue to become empty.
   */
  void await() const {
    auto current_size = std::size_t{};
    while ((current_size = size_.load(std::memory_order_relaxed)) != 0) {
      size_.wait(current_size, std::memory_order_relaxed);
    }
  }

  /* Pushes t to the queue. Blocks if queue is full.
   */
  void push(T t) {
    size_.wait(buffer_size, std::memory_order_relaxed);

    buffer_[back_index_++ % buffer_size] = std::move(t);
    size_.fetch_add(1, std::memory_order_release);
    size_.notify_all();
  }

  /* Removes the front element from the queue. Results in undefined behavior if
   * called on an empty queue, or while other threads are accessing the front of
   * the queue (through either front() or pop()). The intended use is for all
   * consumer threads to call front() repeatedly until they get a value, then
   * synchronize before a single thread calls pop().
   */
  void pop() {
    ++front_index_;
    size_.fetch_sub(1, std::memory_order_relaxed);
    size_.notify_one();
  }

 private:
  static constexpr auto cache_line_size =
      std::hardware_destructive_interference_size;
  alignas(cache_line_size) std::array<T, buffer_size> buffer_{};
  alignas(cache_line_size) std::atomic<std::size_t> size_{};
  alignas(cache_line_size) std::size_t front_index_{};
  alignas(cache_line_size) std::size_t back_index_{};
};

struct Task {
  std::function<void(std::size_t, std::size_t)> operation{};
  std::size_t n_items{};
};

class TPE4 {
 public:
  explicit TPE4(std::size_t n_threads)
      : task_ready_flags_(n_threads),
        n_running_threads_{static_cast<int>(n_threads)} {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      threads_.emplace_back([this, thread_id, n_threads] {
        SCOPED_LOG("thread ", thread_id);
        auto cached_n_items = std::size_t{};
        auto thread_begin_idx = std::size_t{};
        auto thread_end_idx = std::size_t{};
        auto task = decltype(task_queue_.front()){};
        while (true) {
          if (thread_id == 0) {
            n_running_threads_.store(n_threads, std::memory_order_relaxed);
            task = task_queue_.front();
            for (auto& f : task_ready_flags_) {
              f.test_and_set(std::memory_order_release);
            }
            for (auto& f : task_ready_flags_) {
              f.notify_one();
            }
          } else {
            task_ready_flags_[thread_id].wait(false, std::memory_order_acquire);
            task = task_queue_.front();
            task_ready_flags_[thread_id].clear();
          }

          if (task.operation == nullptr) {
            FUNCTION_LOG("thread ", thread_id, " stopping work.");
            break;  // stop-work issued.
          }
          if (cached_n_items != task.n_items) {
            cached_n_items = task.n_items;
            auto const items_per_thread =
                (task.n_items + n_threads - 1) / n_threads;
            thread_begin_idx = thread_id * items_per_thread;
            thread_end_idx =
                std::min((thread_id + 1) * items_per_thread, task.n_items);
          }
          FUNCTION_LOG("thread ", thread_id, " executing task.");
          task.operation(thread_begin_idx, thread_end_idx);
          auto remaining_threads =
              n_running_threads_.fetch_sub(1, std::memory_order_release) - 1;

          if (thread_id == 0) {
            while ((remaining_threads = n_running_threads_.load(
                        std::memory_order_acquire)) != 0) {
              n_running_threads_.wait(remaining_threads,
                                      std::memory_order_relaxed);
            }
            task_queue_.pop();
          } else {
            if (remaining_threads == 0) {
              n_running_threads_.notify_one();
            }
          }
        }
        if (n_running_threads_.fetch_sub(1, std::memory_order_acquire) == 1) {
          FUNCTION_LOG("thread ", thread_id, " popping task.");
          task_queue_.pop();
        }
      });
    }
  }

  ~TPE4() {
    SCOPED_LOG();
    task_queue_.push({});  // null task indicates stop-work.
  }

  auto synchronize() const {
    SCOPED_LOG();
    task_queue_.await();
  }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    SCOPED_LOG();
    task_queue_.push({[... args = std::move(args)](std::size_t thread_begin_idx,
                                                   std::size_t thread_end_idx) {
                        for (auto i = thread_begin_idx; i < thread_end_idx;
                             ++i) {
                          kernel(i, args...);
                        }
                      },
                      n_items});
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
  MultiReaderQ<Task, 8> task_queue_{};
  std::vector<std::atomic_flag> task_ready_flags_{};
  std::atomic_int n_running_threads_{};
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
