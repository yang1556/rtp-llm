#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <functional>

namespace rtp_llm {

/*----------------------------------------------- P2PConnectorAsyncMatchContext
 * -------------------------------------------------*/
size_t P2PConnectorAsyncMatchContext::matchedBlockCount() const {
    auto& layer_block_ids = resource_->layerBlocks();
    if (!layer_block_ids.empty() && layer_block_ids.at(0)) {
        return layer_block_ids.at(0)->blocksNum();
    }
    return 0;
}

bool P2PConnectorAsyncMatchContext::done() const {
    return true;
}

bool P2PConnectorAsyncMatchContext::success() const {
    return true;
}

/*----------------------------------------------- P2PConnectorAsyncReadContext
 * -------------------------------------------------*/
bool P2PConnectorAsyncReadContext::done() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return done_;
}

bool P2PConnectorAsyncReadContext::success() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return success_;
}

void P2PConnectorAsyncReadContext::waitDone() {
    std::unique_lock<std::mutex> lock(state_mutex_);
    done_cv_.wait(lock, [this]() { return done_; });
}

void P2PConnectorAsyncReadContext::checkDone() {
    if (done()) {
        return;
    }
    if (transfer_not_done_hold_pending_.load(std::memory_order_acquire)) {
        const int64_t until_ms = transfer_not_done_hold_until_ms_.load(std::memory_order_relaxed);
        if (currentTimeMs() >= until_ms) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!done_) {
                done_ = true;
            }
            transfer_not_done_hold_pending_.store(false, std::memory_order_release);
            transfer_not_done_hold_until_ms_.store(0, std::memory_order_relaxed);
            done_cv_.notify_all();
        }
        return;
    }
    if (!tp_sync_result_->done()) {
        tp_sync_result_->checkDone();
    }
    if (!server_call_result_->done()) {
        server_call_result_->checkDone();
    }
    const bool both_done = tp_sync_result_->done() && server_call_result_->done();
    if (!both_done) {
        return;
    }

    const bool  success    = tp_sync_result_->success() && server_call_result_->success();
    ErrorCode   error_code = ErrorCode::NONE_ERROR;
    std::string error_message;
    if (!success) {
        if (tp_sync_result_->done() && !tp_sync_result_->success()) {
            error_code    = tp_sync_result_->errorCode();
            error_message = tp_sync_result_->errorMessage();
        } else if (server_call_result_->done() && !server_call_result_->success()) {
            error_code    = server_call_result_->error_code;
            error_message = server_call_result_->error_message;
        }
    }

    if (!success && error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE
        && transfer_not_done_hold_ms_ > 0) {
        transfer_not_done_hold_until_ms_.store(currentTimeMs() + transfer_not_done_hold_ms_, std::memory_order_relaxed);
        transfer_not_done_hold_pending_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            success_       = false;
            error_code_    = error_code;
            error_message_ = std::move(error_message);
        }
        collector_->success                  = false;
        collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
        collector_->tp_sync_cost_time_us     = tp_sync_result_->totalCostTimeUs();
        collector_->server_call_cost_time_us = server_call_result_->totalCostTimeUs();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        done_       = true;
        success_    = success;
        error_code_ = error_code;
        error_message_.assign(error_message);
        done_cv_.notify_all();
    }
    collector_->success                  = success_;
    collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
    collector_->tp_sync_cost_time_us     = tp_sync_result_->totalCostTimeUs();
    collector_->server_call_cost_time_us = server_call_result_->totalCostTimeUs();
}

ErrorInfo P2PConnectorAsyncReadContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ErrorInfo(error_code_, error_message_);
}

bool P2PConnectorAsyncReadContext::needCancel() const {
    if (transfer_not_done_hold_pending_.load()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return false;
        }
    }
    if (tp_sync_result_->done() && !tp_sync_result_->success()) {
        return true;
    }
    if (server_call_result_->done() && !server_call_result_->success()) {
        return true;
    }
    return false;
}

void P2PConnectorAsyncReadContext::cancel(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    std::string unique_key = uniqueKey();

    if (!server_call_result_->done()) {
        server_call_result_->cancel();
    }

    // 如果 tp_sync_result_ 未完成，通过 P2PBroadcastClient 发送 CANCEL 请求
    if (!tp_sync_result_->done() && tp_broadcast_client && !cancel_result_) {
        cancel_result_ = tp_broadcast_client->cancel(unique_key, P2PConnectorBroadcastType::CANCEL_READ);
    }
    if (cancel_result_ && !cancel_result_->done()) {
        cancel_result_->checkDone();
    }
}

/*----------------------------------------------- P2PConnectorAsyncWriteByLayerContext
 * -------------------------------------------------*/
void P2PConnectorAsyncWriteByLayerContext::waitDone() {
    // done() is always true, no blocking
}

bool P2PConnectorAsyncWriteByLayerContext::done() const {
    return true;
}

bool P2PConnectorAsyncWriteByLayerContext::success() const {
    return true;
}

/*----------------------------------------------- P2PConnectorAsyncReadContextChecker
 * -------------------------------------------------*/
P2PConnectorAsyncReadContextChecker::~P2PConnectorAsyncReadContextChecker() {
    stop();
}

bool P2PConnectorAsyncReadContextChecker::init(const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                               const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    metrics_reporter_    = metrics_reporter;
    tp_broadcast_client_ = tp_broadcast_client;
    check_done_thread_ =
        autil::LoopThread::createLoopThread(std::bind(&P2PConnectorAsyncReadContextChecker::checkOnce, this),
                                            5 * 1000,  // 5ms
                                            "P2PConnectorAsyncReadContextCheckerThread");
    if (!check_done_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorAsyncReadContextChecker init failed: check_done_thread is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorAsyncReadContextChecker init success");
    return true;
}

void P2PConnectorAsyncReadContextChecker::stop() {
    if (check_done_thread_) {
        check_done_thread_->stop();
        check_done_thread_.reset();
    }
}

void P2PConnectorAsyncReadContextChecker::addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (!context) {
        return;
    }
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    async_contexts_.push_back(context);
}

size_t P2PConnectorAsyncReadContextChecker::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    return async_contexts_.size();
}

void P2PConnectorAsyncReadContextChecker::checkOnce() {
    int64_t start_time_us = currentTimeUs();

    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    for (auto& async_context : async_contexts_) {
        async_context->checkDone();
        // 检查是否需要取消另一个未完成的请求
        if (async_context->needCancel()) {
            RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContextChecker checkOnce: needCancel, unique_key: %s",
                              async_context->uniqueKey().c_str());
            async_context->cancel(tp_broadcast_client_);
        }
    }
    for (auto& async_context : async_contexts_) {
        if (async_context->done() && !async_context->success()) {
            auto error = async_context->errorInfo();
            RTP_LLM_LOG_WARNING(
                "P2PConnectorAsyncReadContextChecker checkOnce: async read failed, unique_key: %s, error: %s",
                async_context->uniqueKey().c_str(),
                error.ToString().c_str());
        }
    }

    async_contexts_.erase(
        std::remove_if(async_contexts_.begin(),
                       async_contexts_.end(),
                       [](const std::shared_ptr<P2PConnectorAsyncReadContext>& async_context) -> bool {
                           return async_context->done();
                       }),
        async_contexts_.end());

    if (metrics_reporter_) {
        auto collector                     = std::make_shared<DecodeSchedulerStatusMetricsCollector>();
        collector->check_once_cost_time_us = currentTimeUs() - start_time_us;
        collector->inflight_context_count  = async_contexts_.size();
        metrics_reporter_->report<P2PConnectorMetrics, DecodeSchedulerStatusMetricsCollector>(nullptr, collector.get());
    }
}

}  // namespace rtp_llm