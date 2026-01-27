#pragma once

#include <cstddef>
#include <functional>
#include <latch>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "ParallelExecutor.hpp"

namespace ParODE {
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
}  // namespace ParODE
