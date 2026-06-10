#pragma once

#include <atomic>
#include <barrier>
#include <cstddef>
#include <functional>
#include <iterator>
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
  /* Non-destructively reads the front element of the queue. Returns
   * std::nullopt if the queue is empty.
   */
  auto front() const -> std::optional<T> {
    if (size_.load(std::memory_order_acquire) == 0) {
      return std::nullopt;
    }
    return buffer_[front_index_ % buffer_size];
  }

  /* Returns whether the queue is empty. It is intended to be called from the
   * producer thread to confirm that all of the elements pushed to the queue
   * have been consumed. It should not be called from threads other than the
   * producer thread.
   */
  bool empty() const { return size_.load(std::memory_order_relaxed) == 0; }

  /* Pushes t to the queue and returns true, or returns false if the queue is
   * full.
   */
  bool push(T const& t) {
    if (size_.load(std::memory_order_relaxed) >= buffer_size) {
      return false;
    }

    buffer_[back_index_++ % buffer_size] = t;
    size_.fetch_add(1, std::memory_order_release);
    return true;
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
  }

 private:
  std::array<T, buffer_size> buffer_{};
  std::atomic<std::size_t> size_{};
  std::size_t front_index_{};
  std::size_t back_index_{};
};

struct Task {
  std::function<void(std::size_t, std::size_t)> operation{};
  std::size_t n_items{};
};

template <typename Q>
struct Popper {
  Q& queue;
  void operator()() noexcept { queue.pop(); }
};

template <typename Predicate>
bool busy_wait(std::stop_token stop_token, Predicate pred) {
  using namespace std::chrono_literals;
  // auto sleep_time = 1ns;
  for (auto trial = 0; not stop_token.stop_requested(); ++trial) {
    if (pred()) {
      return true;
    }
    if (trial == 8) {
      trial = 0;
      std::this_thread::yield();
    }
    // if (trial == 16) {
    //   trial = 0;
    //   std::this_thread::sleep_for(sleep_time);
    //   if (sleep_time < 1024ns) {
    //     sleep_time *= 2;
    //   }
    // }
  }
  return false;
}

class TPE3 {
 public:
  explicit TPE3(std::size_t n_threads)
      : sync_point_{static_cast<int>(n_threads), Popper{task_queue_}} {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      threads_.emplace_back(
          [this, thread_id, n_threads](std::stop_token stop_token) {
            SCOPED_LOG("thread ", thread_id);
            auto cached_n_items = std::size_t{};
            auto thread_begin_idx = std::size_t{};
            auto thread_end_idx = std::size_t{};
            auto task = decltype(task_queue_.front()){};
            while (true) {
              if (not busy_wait(stop_token, [this, &task] {
                    return (task = task_queue_.front()).has_value();
                  })) {
                break;
              }
              if (cached_n_items != task->n_items) {
                cached_n_items = task->n_items;
                auto const items_per_thread =
                    (task->n_items + n_threads - 1) / n_threads;
                thread_begin_idx = thread_id * items_per_thread;
                thread_end_idx =
                    std::min((thread_id + 1) * items_per_thread, task->n_items);
              }
              task->operation(thread_begin_idx, thread_end_idx);
              sync_point_.arrive_and_wait();
            }
            sync_point_.arrive_and_drop();
          },
          stop_source_.get_token());
    }
  }

  ~TPE3() {
    SCOPED_LOG();
    stop_source_.request_stop();
  }

  auto synchronize() const {
    SCOPED_LOG();
    busy_wait(stop_source_.get_token(), [this] { return task_queue_.empty(); });
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
    busy_wait(stop_source_.get_token(), [this, task = std::move(task)] {
      return task_queue_.push(task);
    });
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
  std::barrier<Popper<decltype(task_queue_)>> sync_point_;
  std::stop_source stop_source_{};
  std::vector<std::jthread> threads_{};
};

using AsyncQueueThreadPoolExecutor = TPE3;

// Template version for testing purposes.
template <std::size_t num_threads>
struct AsyncQueueThreadPoolTemplateExecutor : AsyncQueueThreadPoolExecutor {
  AsyncQueueThreadPoolTemplateExecutor()
      : AsyncQueueThreadPoolExecutor(num_threads) {}
};
}  // namespace ParX
