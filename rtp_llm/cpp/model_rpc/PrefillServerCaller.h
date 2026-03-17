#pragma once

#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include <shared_mutex>
#include <unordered_map>

namespace rtp_llm {

class PrefillServerCaller {
public:
    explicit PrefillServerCaller(const std::string& process_id);
    ~PrefillServerCaller() = default;

    // 调用 Prefill 服务器
    std::shared_ptr<PrefillServerCallerContext> callPrefill(const GenerateInputPB* request,
                                                            const std::string&     ip,
                                                            uint32_t               port,
                                                            const std::string&     unique_key,
                                                            int64_t                deadline_us);

    grpc::Status callPrefill(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer);

    /// @brief Get prefill's tp_size via GetPeerInfo RPC, with caching per endpoint.
    /// Returns 1 on failure (fallback to symmetric TP assumption).
    int getPrefillTpSize(const std::string& ip, uint32_t port);

private:
    std::shared_ptr<RPCPool> rpc_pool_;
    std::string              process_id_;

    mutable std::shared_mutex                   prefill_tp_cache_mutex_;
    std::unordered_map<std::string, int> prefill_tp_cache_;
};

}  // namespace rtp_llm
