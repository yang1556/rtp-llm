#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

struct MHAKVCacheSpec: public KVCacheSpec {
    uint32_t size_per_head;

    MHAKVCacheSpec() = default;

    MHAKVCacheSpec(const AttentionConfigs& attn_config, const ParallelismConfig& parallelism_config) {
        type      = KVCacheSpecType::MultiHeadAttention;
        layer_num = 1;  // Will be set by caller

        // TODO(xinfei.sxf):
        // 这里的head_num_kv分配逻辑需要和ModelConfig::getAttentionConfigs里保持一致，目前这里还是单独计算的
        /*
        KV head partitioning logic for tensor parallelism:
        Case 1: If kv_head_num % tp_size == 0,
            then each rank gets kv_head_num / tp_size KV heads.

        Case 2: If kv_head_num % tp_size != 0,
            then we take the greatest common divisor:
                gcd = GCD(kv_head_num, tp_size),
            and each rank gets kv_head_num / gcd KV heads.
        */
        local_head_num_kv = static_cast<uint32_t>(
            (attn_config.kv_head_num % parallelism_config.get_attn_tp_size() == 0) ?
                attn_config.kv_head_num / parallelism_config.get_attn_tp_size() :
                attn_config.kv_head_num / std::gcd(attn_config.kv_head_num, parallelism_config.get_attn_tp_size()));
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);
        size_per_head      = static_cast<uint32_t>(attn_config.size_per_head);
    }

    // TODO(xinfei.sxf) 下面的函数名字统一掉
    size_t block_size() const override {
        return 2 * local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t k_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
    }

    size_t block_size_bytes() const override {
        return block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t k_block_size_bytes() const override {
        return k_block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size_bytes() const override {
        return v_block_size() * rtp_llm::getTypeSize(dtype);
    }

    // Scale-related methods for MHA (only MHA supports scales for now)
    size_t scale_size_per_block() const {
        // For INT8 or FP8, we need scales for both K and V
        if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
            return 2 * local_head_num_kv * seq_size_per_block;  // K and V scales
        }
        return 0;  // No scales for other data types
    }

    size_t scale_size_bytes_per_block() const {
        return scale_size_per_block() * sizeof(float);
    }

    size_t scale_block_size_bytes() const override {
        return scale_size_bytes_per_block();
    }

    size_t k_scale_block_size_bytes() const override {
        return scale_size_bytes_per_block() / 2;
    }

    size_t v_scale_block_size_bytes() const override {
        return scale_size_bytes_per_block() / 2;
    }

    // Static helper function to split KV partition bytes for MHA
    static KVPartitionBytes splitKVPartitionBytes(size_t      full_block_bytes,
                                                  size_t      k_block_bytes,
                                                  size_t      v_block_bytes,
                                                  int         heads,
                                                  int         partition_count,
                                                  int         partition_id,
                                                  const char* debug_name) {
        if (heads < partition_count) {
            RTP_LLM_CHECK_WITH_INFO(partition_count % heads == 0,
                                    "partition_count must be divisible by heads (%s): partition_count=%d heads=%d",
                                    debug_name,
                                    partition_count,
                                    heads);
            auto partition_heads = partition_count / heads;
            partition_count      = heads;
            partition_id         = partition_id / partition_heads;
        }
        RTP_LLM_CHECK_WITH_INFO(partition_count > 0, "partition_count must be > 0");
        RTP_LLM_CHECK_WITH_INFO(partition_id >= 0 && partition_id < partition_count,
                                "partition_id out of range: %d / %d",
                                partition_id,
                                partition_count);
        RTP_LLM_CHECK_WITH_INFO(heads > 0, "heads must be > 0, got=%d (%s)", heads, debug_name);
        RTP_LLM_CHECK_WITH_INFO(k_block_bytes % static_cast<size_t>(heads) == 0,
                                "k_block_bytes must be divisible by heads (%s): k_partition=%zu heads=%d",
                                debug_name,
                                k_block_bytes,
                                heads);
        RTP_LLM_CHECK_WITH_INFO(v_block_bytes % static_cast<size_t>(heads) == 0,
                                "v_block_bytes must be divisible by heads (%s): v_partition=%zu heads=%d",
                                debug_name,
                                v_block_bytes,
                                heads);

        const size_t k_partition_bytes_per_id = k_block_bytes / static_cast<size_t>(partition_count);
        const size_t v_partition_bytes_per_id = v_block_bytes / static_cast<size_t>(partition_count);

        const size_t k_partition_off = static_cast<size_t>(partition_id) * k_partition_bytes_per_id;
        const size_t v_partition_off = k_block_bytes + static_cast<size_t>(partition_id) * v_partition_bytes_per_id;
        const size_t k_partition_sz  = k_partition_bytes_per_id;
        const size_t v_partition_sz  = v_partition_bytes_per_id;
        return {k_partition_off, k_partition_sz, v_partition_off, v_partition_sz};
    }

    std::string debugString(size_t indent = 0) const override {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << commonDebugString(indent);
        os << indent1 << "size_per_head=" << size_per_head << "\n";
        os << indent1 << "scale_size_per_block=" << scale_size_per_block() << "\n";
        os << indent1 << "scale_size_bytes_per_block=" << scale_size_bytes_per_block() << "\n";
        os << indent1 << "scale_block_size_bytes=" << scale_block_size_bytes() << "\n";
        os << indent1 << "k_scale_block_size_bytes=" << k_scale_block_size_bytes() << "\n";
        os << indent1 << "v_scale_block_size_bytes=" << v_scale_block_size_bytes() << "\n";
        return os.str();
    }
};

}  // namespace rtp_llm