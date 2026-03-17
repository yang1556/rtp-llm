#pragma once

#include <cstdint>

namespace rtp_llm {
namespace transfer {

struct TransferBackendConfig {
    bool    cache_store_rdma_mode               = false;
    int64_t rdma_transfer_wait_timeout_ms       = 180 * 1000;
    int     messager_io_thread_count            = 2;
    int     messager_worker_thread_count        = 16;
    int     rdma_max_block_pairs_per_connection = 0;
    int64_t cache_store_listen_port             = 0;
    int cache_store_tcp_anet_rpc_thread_num = 3;
    int cache_store_tcp_anet_rpc_queue_num  = 100;
};

}  // namespace transfer
}  // namespace rtp_llm
