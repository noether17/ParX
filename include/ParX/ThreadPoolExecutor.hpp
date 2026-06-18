#pragma once

#include <atomic>
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
/* Waits for predicate(testable) to return true by repeatedly checking the
 * return value. Spins num_spins times, pausing on each spin, before yielding
 * the thread.
 */
template <int num_spins = 16>
void busy_wait_until(auto&& testable, auto&& predicate) {
  for (auto spin_count = 0; not predicate(testable); ++spin_count) {
    __builtin_ia32_pause();
    if (spin_count == num_spins - 1) {
      spin_count = 0;
      std::this_thread::yield();
    }
  }
}

/* Checks predicate on atomic_value using acquire memory order. If the predicate
 * returns false, busy waits until it returns true using relaxed memory order
 * before the final check using acquire memory order. */
template <int num_spins = 16>
void busy_wait_until(auto&& atomic_value, auto&& predicate)
  requires requires(decltype(atomic_value) v, std::memory_order order) {
    v.load(order);
  }
{
  while (not predicate(atomic_value.load(std::memory_order_acquire))) {
    busy_wait_until<num_spins>(
        [&] { return atomic_value.load(std::memory_order_relaxed); },
        [&](auto&& t_loader) { return predicate(t_loader()); });
  }
}

template <typename T, std::size_t buffer_size>
class LockFreeQueue {
 public:
  LockFreeQueue()
      : buffer_{static_cast<T*>(std::malloc(buffer_size * sizeof(T)))} {}
  LockFreeQueue(LockFreeQueue const&) = delete;
  LockFreeQueue(LockFreeQueue&&) = delete;
  auto& operator=(LockFreeQueue const&) = delete;
  auto& operator=(LockFreeQueue&&) = delete;
  ~LockFreeQueue() { std::free(buffer_); }

  auto pop() {
    auto element = std::move(buffer_[front_index_ % buffer_size]);
    buffer_[front_index_ % buffer_size].~T();
    ++front_index_;
    size_.fetch_sub(1, std::memory_order_relaxed);
    return element;
  }

  void push(T t) {
    new (&buffer_[back_index_ % buffer_size]) T(std::move(t));
    ++back_index_;
    size_.fetch_add(1, std::memory_order_release);
  }

  void wait_until_empty() const {
    busy_wait_until(size_, [](auto n) { return n == 0; });
  }

  void wait_while_empty() const {
    busy_wait_until(size_, [](auto n) { return n != 0; });
  }

  void wait_while_full() const {
    busy_wait_until(size_, [](auto n) { return n < buffer_size; });
  }

 private:
  T* const buffer_{};
  alignas(std::hardware_destructive_interference_size)
      std::atomic<std::size_t> size_{};
  alignas(std::hardware_destructive_interference_size) std::size_t
      front_index_{};
  alignas(std::hardware_destructive_interference_size) std::size_t
      back_index_{};
};

struct Task {
  std::function<void(std::size_t, std::size_t)> operation{};
  std::size_t n_items{};
};

template <std::size_t queue_size = 16>
class ThreadPoolExecutor {
 public:
  explicit ThreadPoolExecutor(std::size_t n_threads)
      : task_ready_flags_(n_threads) {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      threads_.emplace_back([this, thread_id, n_threads] {
        SCOPED_LOG("thread ", thread_id);
        auto cached_n_items = std::size_t{};
        auto thread_begin_idx = std::size_t{};
        auto thread_end_idx = std::size_t{};
        while (true) {
          if (thread_id == 0) {
            busy_wait_until(n_running_threads_, [](auto n) { return n == 0; });
            task_queue_.wait_while_empty();
            n_running_threads_.store(n_threads);
            active_task_ = task_queue_.pop();
            for (auto& f : task_ready_flags_) {
              f.store(true, std::memory_order_release);
            }
          } else {
            busy_wait_until(task_ready_flags_[thread_id],
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

  ThreadPoolExecutor(ThreadPoolExecutor const&) = delete;
  ThreadPoolExecutor(ThreadPoolExecutor&&) = delete;
  auto& operator=(ThreadPoolExecutor const&) = delete;
  auto& operator=(ThreadPoolExecutor&&) = delete;

  ~ThreadPoolExecutor() {
    SCOPED_LOG();
    task_queue_.wait_while_full();
    task_queue_.push({});  // Null task indicates stop-work.
  }

  auto synchronize() const {
    SCOPED_LOG();
    task_queue_.wait_until_empty();
    busy_wait_until(n_running_threads_, [](auto n) { return n == 0; });
  }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    SCOPED_LOG();
    task_queue_.wait_while_full();
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
  alignas(std::hardware_destructive_interference_size)
      LockFreeQueue<Task, queue_size> task_queue_{};
  alignas(std::hardware_destructive_interference_size) Task active_task_{};
  alignas(std::hardware_destructive_interference_size) std::atomic_int
      n_running_threads_{};
  struct alignas(std::hardware_destructive_interference_size) Flag
      : std::atomic_bool {};
  alignas(std::hardware_destructive_interference_size)
      std::vector<Flag> task_ready_flags_{};
  std::vector<std::jthread> threads_{};
};

// Template version for testing purposes.
template <std::size_t num_threads, std::size_t queue_size = 16>
struct ThreadPoolTemplateExecutor : ThreadPoolExecutor<queue_size> {
  ThreadPoolTemplateExecutor() : ThreadPoolExecutor<queue_size>(num_threads) {}
};
}  // namespace ParX
