#pragma once

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "ParX/ParallelExecutor.hpp"
#include "ParX/util/Logging.hpp"

namespace ParX {
template <typename Predicate>
bool spinlock(std::stop_token& stop_token, Predicate pred) {
  for (auto trial = 0; not stop_token.stop_requested(); ++trial) {
    if (pred()) {
      return true;
    }
    if (trial == 8) {
      trial = 0;
      std::this_thread::yield();
    }
  }
  return false;
}

template <typename T>
class MultiReaderQueue {
 public:
  auto front() const -> std::optional<T> {
    auto const log = ScopedLogger{};
    FUNCTION_LOG("entry point.");
    auto read_lock = std::shared_lock{mx_};
    read_cv_.wait(read_lock,
                  [this] { return data_.has_value() or stop_flag_; });
    if (stop_flag_) {
      FUNCTION_LOG("stopping.");
      return std::nullopt;
    }
    FUNCTION_LOG("returning front value.");
    return data_;
  }

  auto empty() const {
    FUNCTION_LOG("entry point.");
    auto read_lock = std::shared_lock{mx_};
    FUNCTION_LOG("returning.");
    return not data_.has_value();
  }

  auto push(T t) {
    FUNCTION_LOG("entry point.");
    {
      auto write_lock = std::unique_lock{mx_};
      empty_cv_.wait(write_lock,
                     [this] { return not data_.has_value() or stop_flag_; });
      if (stop_flag_) {
        FUNCTION_LOG("stopping.");
        return;
      }
      data_ = std::move(t);
      FUNCTION_LOG("pushed value.");
    }
    read_cv_.notify_all();
  }

  auto pop() {
    FUNCTION_LOG("entry point.");
    {
      auto write_lock = std::unique_lock{mx_};
      data_.reset();
      FUNCTION_LOG("popped value.");
    }
    empty_cv_.notify_all();
  }

  auto wait() const {
    FUNCTION_LOG("entry point.");
    {
      auto read_lock = std::shared_lock{mx_};
      empty_cv_.wait(read_lock,
                     [this] { return not data_.has_value() or stop_flag_; });
    }
    if (stop_flag_) {
      FUNCTION_LOG("stopping.");
      return;
    }
    FUNCTION_LOG("completed.");
  }

  auto stop() {
    FUNCTION_LOG("entry point.");
    stop_flag_ = true;
    FUNCTION_LOG("set stop flag.");
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
        auto cached_n_items = std::size_t{};
        auto thread_begin_idx = std::size_t{};
        auto thread_end_idx = std::size_t{};
        while (true) {
          FUNCTION_LOG("thread ", thread_id, " starting loop iteration.");
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
        FUNCTION_LOG("thread ", thread_id, " completed.");
      });
    }
  }

  ~TPE2() {
    FUNCTION_LOG("entry point.");
    task_queue_.stop();
  }

  auto synchronize() const {
    FUNCTION_LOG("entry point.");
    task_queue_.wait();
  }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    FUNCTION_LOG("entry point.");
    task_queue_.push({[... args = std::move(args)](std::size_t thread_begin_idx,
                                                   std::size_t thread_end_idx) {
                        for (auto i = thread_begin_idx; i < thread_end_idx;
                             ++i) {
                          kernel(i, args...);
                        }
                      },
                      n_items});
    FUNCTION_LOG("pushed task to queue.");
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  static constexpr void transform_reduce_kernel(std::size_t thread_id,
                                                T* thread_partial_results,
                                                std::size_t n_items,
                                                std::size_t n_items_per_thread,
                                                TransformArgs... transform_args)
    requires(Reduction<reduce, T> and Transform<transform, T, TransformArgs...>)
  {
    FUNCTION_LOG("entry point.");
    auto thread_partial_result = T{};
    for (auto i = thread_id * n_items_per_thread;
         i < (thread_id + 1) * n_items_per_thread and i < n_items; ++i) {
      auto transform_result = transform(i, transform_args...);
      thread_partial_result = reduce(thread_partial_result, transform_result);
    }
    thread_partial_results[thread_id] = thread_partial_result;
    FUNCTION_LOG("completed.");
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, std::size_t n_items,
                        TransformArgs... transform_args)
    requires(Reduction<reduce, T> and Transform<transform, T, TransformArgs...>)
  {
    FUNCTION_LOG("entry point.");
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
    FUNCTION_LOG("completed.");
  }

  constexpr auto n_threads() const { return std::ssize(threads_); }

 private:
  MultiReaderQueue<Task> task_queue_{};
  std::barrier<Popper<decltype(task_queue_)>> sync_point_;
  std::vector<std::jthread> threads_{};
};

template <typename T>
class ThreadSafeQueue {
 public:
  auto front() const -> std::optional<T> {
    std::cout << "ThreadSafeQueue::front(): entry point.\n";
    auto read_lock = std::shared_lock{mx_};
    cv_.wait(read_lock, [this] { return stop_flag_ or not data_.empty(); });
    if (stop_flag_) {
      std::cout << "ThreadSafeQueue::front(): stopping.\n";
      return std::nullopt;
    }
    std::cout << "ThreadSafeQueue::front(): returning front value.\n";
    return data_.front();
  }

  auto empty() const {
    std::cout << "ThreadSafeQueue::empty(): entry point.\n";
    auto read_lock = std::shared_lock{mx_};
    std::cout << "ThreadSafeQueue::empty(): returning.\n";
    return data_.empty();
  }

  auto push(T t) {
    std::cout << "ThreadSafeQueue::push(): entry point.\n";
    {
      auto write_lock = std::unique_lock{mx_};
      data_.push(std::move(t));
    }
    std::cout << "ThreadSafeQueue::push(): pushed value.\n";
    cv_.notify_all();
  }

  auto pop() {
    std::cout << "ThreadSafeQueue::pop(): entry point.\n";
    auto write_lock = std::unique_lock{mx_};
    data_.pop();
    std::cout << "ThreadSafeQueue::pop(): popped value.\n";
    cv_.notify_all();
  }

  auto wait() const {
    std::cout << "ThreadSafeQueue::wait(): entry point.\n";
    auto read_lock = std::shared_lock{mx_};
    cv_.wait(read_lock, [this] {
      auto ss = std::stringstream{};
      ss << "ThreadSafeQueue::wait::lambda(): " << data_.size()
         << " elements in queue.\n";
      std::cout << ss.str();
      return data_.empty();
    });
    std::cout << "ThreadSafeQueue::wait(): finished.\n";
  }

  auto stop() {
    std::cout << "ThreadSafeQueue::stop(): entry point.\n";
    stop_flag_ = true;
    std::cout << "ThreadSafeQueue::stop(): set stop flag.\n";
    cv_.notify_all();
  }

  ~ThreadSafeQueue() {
    std::cout << "ThreadSafeQueue::~ThreadSafeQueue(): entry point.\n";
    stop();
    wait();
    std::cout << "ThreadSafeQueue::~ThreadSafeQueue(): finished.\n";
  }

 private:
  std::queue<T> data_{};
  std::shared_mutex mutable mx_{};
  std::condition_variable_any mutable cv_{};
  std::atomic_bool stop_flag_{};
};

class TPE1 {
 public:
  explicit TPE1(std::size_t n_threads)
      : sync_point_{static_cast<std::ptrdiff_t>(n_threads),
                    TaskPopper{task_, task_mx_, write_cv_}} {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      threads_.emplace_back(
          [this, thread_id, n_threads](std::stop_token stop_token) {
            auto cached_n_items = std::size_t{};
            auto thread_begin = std::size_t{};
            auto thread_end = std::size_t{};
            auto ss = std::stringstream{};
            while (true) {
              //
              // ss = {};
              // ss << "Thread " << thread_id << " starting loop.\n";
              // std::cout << ss.str();
              //
              auto task = [this, stop_token] {
                auto read_lock = std::shared_lock{task_mx_};
                read_cv_.wait(read_lock, [this, stop_token] {
                  return task_.has_value() or stop_token.stop_requested();
                });
                return task_;
              }();
              //
              // ss = {};
              // ss << "Thread " << thread_id << " finished waiting for
              // read.\n"; std::cout << ss.str();
              //
              if (stop_token.stop_requested()) {
                break;
              }
              if (cached_n_items != task->n_items) {
                cached_n_items = task->n_items;
                auto items_per_thread =
                    (cached_n_items + n_threads - 1) / n_threads;
                thread_begin = thread_id * items_per_thread;
                thread_end = std::min((thread_id + 1) * items_per_thread,
                                      cached_n_items);
              }
              task->operation(thread_begin, thread_end);
              //
              // ss = {};
              // ss << "Thread " << thread_id << " finished task.\n";
              // std::cout << ss.str();
              //
              sync_point_.arrive_and_wait();
            }
            //
            // ss = {};
            // ss << "Thread " << thread_id << " completed.\n";
            // std::cout << ss.str();
            //
            sync_point_.arrive_and_drop();
          },
          std::stop_token{stop_source_.get_token()});
    }
  }

  ~TPE1() {
    stop_source_.request_stop();
    read_cv_.notify_all();
  }

  auto synchronize() const {
    //
    // auto ss = std::stringstream{};
    // ss << "synchronize(): entry point.\n";
    // std::cout << ss.str();
    //
    {
      auto read_lock = std::shared_lock{task_mx_};
      write_cv_.wait(read_lock, [this] { return not task_.has_value(); });
    }
    //
    // ss = {};
    // ss << "synchronize(): exit point.\n";
    // std::cout << ss.str();
    //
  }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    //
    // auto ss = std::stringstream{};
    // ss << "call_kernel(): entry point.\n";
    // std::cout << ss.str();
    //
    {
      auto write_lock = std::unique_lock{task_mx_};
      write_cv_.wait(write_lock, [this] { return not task_.has_value(); });
      task_ = {[... args = std::move(args)](std::size_t thread_begin,
                                            std::size_t thread_end) {
                 for (auto i = thread_begin; i < thread_end; ++i) {
                   kernel(i, args...);
                 }
               },
               n_items};
    }
    //
    // ss = {};
    // ss << "call_kernel(): wrote to task_.\n";
    // std::cout << ss.str();
    //
    read_cv_.notify_all();
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  static constexpr void transform_reduce_kernel(int thread_id,
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
    auto const n_threads = std::ssize(threads_);
    auto thread_partial_results = std::vector<T>(n_threads);
    auto n_items_per_thread = (n_items + n_threads - 1) / n_threads;
    call_kernel<
        transform_reduce_kernel<T, reduce, transform, TransformArgs...>>(
        n_threads, thread_partial_results.data(), n_items, n_items_per_thread,
        transform_args...);
    synchronize();
    return std::accumulate(thread_partial_results.begin(),
                           thread_partial_results.end(), init_val, reduce);
  }

  constexpr auto n_threads() const { return std::ssize(threads_); }

 private:
  struct Task {
    std::function<void(std::size_t, std::size_t)> operation{};
    std::size_t n_items{};
  };
  struct TaskPopper {
    std::optional<Task>& task_;
    std::shared_mutex& mx_;
    std::condition_variable_any& write_cv_;

    void operator()() {
      {
        auto write_lock = std::unique_lock{mx_};
        task_.reset();
      }
      write_cv_.notify_all();
    }
  };
  std::optional<Task> task_{};
  std::shared_mutex mutable task_mx_{};
  std::condition_variable_any mutable read_cv_{};
  std::condition_variable_any mutable write_cv_{};
  std::barrier<TaskPopper> sync_point_;
  std::stop_source stop_source_{};
  std::vector<std::jthread> threads_{};
};

class TPE {
 public:
  explicit TPE(std::size_t n_threads)
      : sync_point_{static_cast<int>(n_threads), TaskPopper{task_queue_}} {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      threads_.emplace_back(
          [this, thread_id, n_threads](std::stop_token stop_token) {
            auto cached_n_items = std::size_t{};
            auto thread_begin = std::size_t{};
            auto thread_end = std::size_t{};
            while (not stop_token.stop_requested()) {
              //
              auto ss = std::stringstream{};
              ss << "Thread " << thread_id << " entering loop.\n";
              std::cout << ss.str();
              //
              auto task = task_queue_.front();
              if (not task.has_value()) {
                break;
              }
              if (cached_n_items != task->n_items) {
                cached_n_items = task->n_items;
                auto items_per_thread =
                    (task->n_items + n_threads - 1) / n_threads;
                thread_begin = thread_id * items_per_thread;
                thread_end =
                    std::min((thread_id + 1) * items_per_thread, task->n_items);
              }
              task->operation(thread_begin, thread_end);
              sync_point_.arrive_and_wait();
            }
            sync_point_.arrive_and_drop();
          },
          std::stop_token{stop_source_.get_token()});
    }
  }

  ~TPE() {
    task_queue_.stop();
    stop_source_.request_stop();
  }

  auto synchronize() const { task_queue_.wait(); }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    task_queue_.push({[... args = std::move(args)](std::size_t thread_begin,
                                                   std::size_t thread_end) {
                        for (auto i = thread_begin; i < thread_end; ++i) {
                          kernel(i, args...);
                        }
                      },
                      n_items});
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  static constexpr void transform_reduce_kernel(int thread_id,
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
    auto const n_threads = std::ssize(threads_);
    auto thread_partial_results = std::vector<T>(n_threads);
    auto n_items_per_thread = (n_items + n_threads - 1) / n_threads;
    call_kernel<
        transform_reduce_kernel<T, reduce, transform, TransformArgs...>>(
        n_threads, thread_partial_results.data(), n_items, n_items_per_thread,
        transform_args...);
    synchronize();
    return std::accumulate(thread_partial_results.begin(),
                           thread_partial_results.end(), init_val, reduce);
  }

  constexpr auto n_threads() const { return std::ssize(threads_); }

 private:
  struct Task {
    std::function<void(std::size_t, std::size_t)> operation{};
    std::size_t n_items{};
  };

  struct TaskPopper {
    ThreadSafeQueue<Task>& task_queue_;
    void operator()() const noexcept { task_queue_.pop(); }
  };

  ThreadSafeQueue<Task> task_queue_{};
  std::stop_source stop_source_{};
  std::barrier<TaskPopper> sync_point_;
  std::vector<std::jthread> threads_{};
};

using ThreadPoolExecutor = TPE2;

/*
class ThreadPoolExecutor {
 public:
  explicit ThreadPoolExecutor(std::size_t n_threads)
      : m_task_ready_flags(n_threads) {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      m_threads.emplace_back(
          [this, thread_id, n_threads](std::stop_token stop_token) {
            auto old_n_items = std::size_t{};
            auto thread_begin = std::size_t{};
            auto thread_end = std::size_t{};
            while (true) {
              if (not spinlock(stop_token, [this, thread_id] {
                    return m_task_ready_flags[thread_id].load();
                  })) {
                return;
              }
              m_task_ready_flags[thread_id] = false;

              if (auto current_n_items = m_n_items.load();
                  current_n_items != old_n_items) {
                old_n_items = current_n_items;
                auto items_per_thread =
                    (current_n_items + n_threads - 1) / n_threads;
                thread_begin = thread_id * items_per_thread;
                thread_end = std::min((thread_id + 1) * items_per_thread,
                                      current_n_items);
              }

              m_task(thread_begin, thread_end);
            }
          },
          std::stop_token{m_stop_source.get_token()});
    }
  }

  ~ThreadPoolExecutor() { m_stop_source.request_stop(); }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    auto latch = std::latch{std::ssize(m_threads)};
    m_n_items = n_items;
    m_task = [&latch, ... args = std::move(args)](int thread_begin,
                                                  int thread_end) {
      for (auto i = thread_begin; i < thread_end; ++i) {
        kernel(i, args...);
      }
      latch.count_down();
    };
    for (auto& flag : m_task_ready_flags) {
      flag = true;
    }
    latch.wait();
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  static constexpr void transform_reduce_kernel(int thread_id,
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
    auto const n_threads = std::ssize(m_threads);
    auto thread_partial_results = std::vector<T>(n_threads);
    auto n_items_per_thread = (n_items + n_threads - 1) / n_threads;
    call_kernel<
        transform_reduce_kernel<T, reduce, transform, TransformArgs...>>(
        n_threads, thread_partial_results.data(), n_items, n_items_per_thread,
        transform_args...);
    return std::accumulate(thread_partial_results.begin(),
                           thread_partial_results.end(), init_val, reduce);
  }

  constexpr auto n_threads() const { return std::ssize(m_threads); }

 private:
  std::stop_source m_stop_source{};
  std::function<void(std::size_t, std::size_t)> m_task{};
  std::vector<std::atomic_bool> m_task_ready_flags{};
  std::atomic_ulong m_n_items{};
  std::vector<std::jthread> m_threads{};
};
*/

// Template version for testing purposes.
template <std::size_t num_threads>
struct ThreadPoolTemplateExecutor : ThreadPoolExecutor {
  ThreadPoolTemplateExecutor() : ThreadPoolExecutor(num_threads) {}
};
}  // namespace ParX
