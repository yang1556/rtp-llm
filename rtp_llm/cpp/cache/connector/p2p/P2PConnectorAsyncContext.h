#pragma once

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/PrefillLoadCaller.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "autil/LoopThread.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace rtp_llm {

/// @brief PD 分离场景下的匹配上下文，始终全量匹配
class P2PConnectorAsyncMatchContext: public AsyncMatchContext {
public:
    P2PConnectorAsyncMatchContext(const KVCacheResourcePtr& resource): resource_(resource) {}
    virtual ~P2PConnectorAsyncMatchContext() {}

public:
    size_t matchedBlockCount() const override;
    bool   done() const override;
    bool   success() const override;
    void   waitDone() override {}

private:
    KVCacheResourcePtr resource_;
};

class P2PConnectorAsyncReadContext: public AsyncContext {
public:
    P2PConnectorAsyncReadContext(const KVCacheResourcePtr&                               resource,
                                 const std::shared_ptr<P2PBroadcastClient::Result>&      tp_sync_result,
                                 const std::shared_ptr<PrefillLoadCaller::Result>&       server_call_result,
                                 const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
                                 int64_t                                                 transfer_not_done_hold_ms):
        resource_(resource),
        tp_sync_result_(tp_sync_result),
        server_call_result_(server_call_result),
        collector_(collector),
        transfer_not_done_hold_ms_(transfer_not_done_hold_ms),
        done_(false),
        success_(false),
        error_code_(ErrorCode::NONE_ERROR) {}
    virtual ~P2PConnectorAsyncReadContext() = default;

public:
    void waitDone() override;
    bool done() const override;
    bool success() const override;

    void checkDone();
    bool needCancel() const;
    void cancel(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);

    std::string uniqueKey() const {
        return tp_sync_result_ ? tp_sync_result_->uniqueKey() : "";
    }

    ErrorInfo errorInfo() const override;

private:
    KVCacheResourcePtr                               resource_;
    std::shared_ptr<P2PBroadcastClient::Result>      tp_sync_result_;
    std::shared_ptr<PrefillLoadCaller::Result>       server_call_result_;
    std::shared_ptr<DecodeSchedulerMetricsCollector> collector_;
    std::shared_ptr<P2PBroadcastClient::Result>      cancel_result_;

    const int64_t transfer_not_done_hold_ms_;

    mutable std::mutex      state_mutex_;
    std::condition_variable done_cv_;
    bool                    done_{false};
    bool                    success_{false};
    ErrorCode               error_code_;
    std::string             error_message_;
    std::atomic<bool>       transfer_not_done_hold_pending_{false};
    std::atomic<int64_t>    transfer_not_done_hold_until_ms_{0};
};

class P2PConnectorAsyncWriteByLayerContext: public AsyncContext {
public:
    P2PConnectorAsyncWriteByLayerContext(const KVCacheResourcePtr& resource): resource_(resource) {}
    virtual ~P2PConnectorAsyncWriteByLayerContext() {}

public:
    void waitDone() override;  // done() always true, no blocking
    bool done() const override;
    bool success() const override;

private:
    KVCacheResourcePtr resource_;
};

/// @brief 后台线程定期检查 in-flight 异步 read 上下文，超时时自动取消
class P2PConnectorAsyncReadContextChecker {
public:
    P2PConnectorAsyncReadContextChecker() = default;
    ~P2PConnectorAsyncReadContextChecker();

public:
    /// @brief 启动后台检查线程
    bool init(const kmonitor::MetricsReporterPtr&        metrics_reporter,
              const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);
    void stop();
    /// @brief 添加需要跟踪的异步 read 上下文
    void   addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context);
    size_t inflightContextCount() const;

private:
    void checkOnce();

private:
    kmonitor::MetricsReporterPtr                               metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>                        tp_broadcast_client_;
    mutable std::mutex                                         async_contexts_mutex_;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> async_contexts_;
    autil::LoopThreadPtr                                       check_done_thread_;
};

}  // namespace rtp_llm