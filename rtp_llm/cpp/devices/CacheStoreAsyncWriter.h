#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>

#include "autil/ThreadPool.h"

namespace rtp_llm {
// Offloads writeCacheStore CPU-heavy work to a single background thread so the
// main thread can keep launching CUDA kernels without stalling.

// **Critical**:
// Thread-safety: init / submit / waitAllDone are expected to be called from the
// same (main) thread.  Only the background worker touches pending_count_ and
// stored_exception_ concurrently; their visibility is guaranteed by the
// acquire-release on pending_count_.

// Strict lifecycle: init() -> submit()* -> waitAllDone() -> init() -> ...
//   - init()        : IDLE -> RUNNING.  Aborts if already RUNNING.
//   - submit()      : enqueue a task.   Aborts if not RUNNING.
//   - waitAllDone() : blocks until all tasks finish, RUNNING -> IDLE.
//                     Aborts if already IDLE.  Re-throws the first background
//                     exception (if any) after transitioning to IDLE.

class CacheStoreAsyncWriter {
public:
    CacheStoreAsyncWriter();
    ~CacheStoreAsyncWriter();

    void init();
    void submit(std::function<void()> task);
    void waitAllDone();

    bool isIdle() const {
        return state_ == State::IDLE;
    }

private:
    enum class State {
        IDLE,
        RUNNING
    };

    autil::ThreadPoolBasePtr thread_pool_;
    std::atomic<int64_t>     pending_count_{0};
    std::mutex               wait_mutex_;
    std::condition_variable  wait_cv_;
    std::exception_ptr       stored_exception_;
    State                    state_{State::IDLE};
};

}  // namespace rtp_llm
