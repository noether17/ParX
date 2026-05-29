#pragma once

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

#include "ParX/ParallelExecutor.hpp"
#include "ParX/util/Logging.hpp"

namespace ParX {
template <typename T>
class MultiReaderQueue {
 public:
  auto front() const -> std::optional<T> {
    SCOPED_LOG();
    auto read_lock = std::shared_lock{mx_};
    read_cv_.wait(read_lock,
                  [this] { return data_.has_value() or stop_flag_; });
    if (stop_flag_) {
      FUNCTION_LOG("stopping.");
      return std::nullopt;
    }
    return data_;
  }

  auto empty() const {
    SCOPED_LOG();
    auto read_lock = std::shared_lock{mx_};
    return not data_.has_value();
  }

  auto push(T t) {
    SCOPED_LOG();
    {
      auto write_lock = std::unique_lock{mx_};
      empty_cv_.wait(write_lock,
                     [this] { return not data_.has_value() or stop_flag_; });
      if (stop_flag_) {
        FUNCTION_LOG("stopping.");
        return;
      }
      data_ = std::move(t);
    }
    read_cv_.notify_all();
  }

  auto pop() {
    SCOPED_LOG();
    {
      auto write_lock = std::unique_lock{mx_};
      data_.reset();
    }
    empty_cv_.notify_all();
  }

  auto wait() const {
    SCOPED_LOG();
    {
      auto read_lock = std::shared_lock{mx_};
      empty_cv_.wait(read_lock,
                     [this] { return not data_.has_value() or stop_flag_; });
    }
    if (stop_flag_) {
      FUNCTION_LOG("stopping.");
      return;
    }
  }

  auto stop() {
    SCOPED_LOG();
    stop_flag_ = true;
    read_cv_.notify_all();
    empty_cv_.notify_all();
  }

 private:
  std::optional<T> data_{};  // std::optional for now; compare performance with
                             // std::queue later.
  std::shared_mutex mutable mx_{};
  std::condition_variable_any mutable read_cv_{};   // signals value available.
  std::condition_variable_any mutable empty_cv_{};  // signals empty data_.
  std::atomic_bool stop_flag_{};
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

class TPE2 {
 public:
  explicit TPE2(std::size_t n_threads)
      : sync_point_{static_cast<int>(n_threads), Popper{task_queue_}} {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      threads_.emplace_back([this, thread_id, n_threads] {
        SCOPED_LOG("thread ", thread_id);
        auto cached_n_items = std::size_t{};
        auto thread_begin_idx = std::size_t{};
        auto thread_end_idx = std::size_t{};
        while (true) {
          auto task = task_queue_.front();
          if (not task.has_value()) {
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
      });
    }
  }

  ~TPE2() {
    SCOPED_LOG();
    task_queue_.stop();
  }

  auto synchronize() const {
    SCOPED_LOG();
    task_queue_.wait();
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
  static constexpr void transform_reduce_kernel(std::size_t thread_id,
                                                T* thread_partial_results,
                                                std::size_t n_items,
                                                std::size_t n_items_per_thread,
                                                TransformArgs... transform_args)
    requires(Reduction<reduce, T> and Transform<transform, T, TransformArgs...>)
  {
    auto thread_partial_result = T{};
    for (auto i = thread_id * n_items_per_thread;
         i < (thread_id + 1) * n_items_per_thread and i < n_items; ++i) {
      auto transform_result = transform(i, transform_args...);
      thread_partial_result = reduce(thread_partial_result, transform_result);
    }
    thread_partial_results[thread_id] = thread_partial_result;
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, std::size_t n_items,
                        TransformArgs... transform_args)
    requires(Reduction<reduce, T> and Transform<transform, T, TransformArgs...>)
  {
    SCOPED_LOG();
    auto const n_threads = std::ssize(threads_);
    auto thread_partial_results =
        std::vector<T>(n_threads);  // TODO: Remove this allocation (likely need
                                    // n_threads to be a template parameter so
                                    // std::array can be used instead).
    auto n_items_per_thread = (n_items + n_threads - 1) / n_threads;
    call_kernel<
        transform_reduce_kernel<T, reduce, transform, TransformArgs...>>(
        n_threads, thread_partial_results.data(), n_items, n_items_per_thread,
        transform_args...);
    synchronize();
    FUNCTION_LOG("synchronized threads.");
    return std::accumulate(std::begin(thread_partial_results),
                           std::end(thread_partial_results), init_val, reduce);
  }

  constexpr auto n_threads() const { return std::ssize(threads_); }

 private:
  MultiReaderQueue<Task> task_queue_{};
  std::barrier<Popper<decltype(task_queue_)>> sync_point_;
  std::vector<std::jthread> threads_{};
};

using AsyncThreadPoolExecutor = TPE2;

// Template version for testing purposes.
template <std::size_t num_threads>
struct AsyncThreadPoolTemplateExecutor : AsyncThreadPoolExecutor {
  AsyncThreadPoolTemplateExecutor() : AsyncThreadPoolExecutor(num_threads) {}
};
}  // namespace ParX
