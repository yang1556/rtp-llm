#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class LayerBlockConverterImpl: public LayerBlockConverter {
public:
    explicit LayerBlockConverterImpl(const std::shared_ptr<KVCacheAllocator>& allocator): allocator_(allocator) {}

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        auto block_infos = allocator_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
        std::vector<BlockInfo> result;
        result.reserve(block_infos.size());
        for (const auto& info : block_infos) {
            if (info.addr != nullptr && info.size_bytes > 0) {
                result.push_back(info);
            }
        }
        return result;
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        auto raw_buffers = allocator_->getAllBuffers();
        std::vector<std::pair<BlockInfo, size_t>> result;
        result.reserve(raw_buffers.size());
        for (auto& [buf_ptr, aligned_size] : raw_buffers) {
            if (!buf_ptr) {
                continue;
            }
            BlockInfo info;
            info.is_cuda      = (buf_ptr->where() == MemoryType::MEMORY_GPU);
            info.addr         = buf_ptr->data();
            info.size_bytes   = buf_ptr->sizeBytes();
            result.push_back({info, aligned_size});
        }
        return result;
    }

private:
    std::shared_ptr<KVCacheAllocator> allocator_;
};

}  // namespace rtp_llm
