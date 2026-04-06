#include "rtp_llm/cpp/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/kernels/block_zero_kernels.h"
#include <cuda_runtime.h>
#include <vector>

using namespace rtp_llm;

class CudaBlockZeroKernelTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        DeviceTestBase::TearDown();
    }
    cudaStream_t stream_ = nullptr;

    struct GpuMem {
        uint8_t* buf   = nullptr;
        void**   bases = nullptr;
        ~GpuMem() {
            cudaFree(buf);
            cudaFree(bases);
        }
    };

    GpuMem setupMemory(size_t layers, size_t blocks, size_t stride, uint8_t fill) {
        GpuMem m;
        size_t layer_bytes = blocks * stride;
        cudaMalloc(&m.buf, layers * layer_bytes);
        cudaMemset(m.buf, fill, layers * layer_bytes);

        std::vector<void*> h_bases(layers);
        for (size_t l = 0; l < layers; ++l)
            h_bases[l] = m.buf + l * layer_bytes;
        cudaMalloc(&m.bases, layers * sizeof(void*));
        cudaMemcpy(m.bases, h_bases.data(), layers * sizeof(void*), cudaMemcpyHostToDevice);
        return m;
    }

    template <typename T>
    T* toDevice(const std::vector<T>& h) {
        T* d = nullptr;
        cudaMalloc(&d, h.size() * sizeof(T));
        cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
        return d;
    }

    std::vector<uint8_t> download(uint8_t* d_ptr, size_t bytes) {
        std::vector<uint8_t> h(bytes);
        cudaMemcpy(h.data(), d_ptr, bytes, cudaMemcpyDeviceToHost);
        return h;
    }

    bool isAllValue(const uint8_t* data, size_t len, uint8_t val) {
        for (size_t i = 0; i < len; ++i)
            if (data[i] != val)
                return false;
        return true;
    }
};

// Zeros block only at new-block boundary: (tokens-1) % seq_size_per_block == 0.
TEST_F(CudaBlockZeroKernelTest, ZerosOnlyNewBlockBoundary) {
    constexpr size_t kLayers         = 2;
    constexpr size_t kBlocks         = 8;
    constexpr size_t kStride         = 256;
    constexpr size_t kSeqPerBlock    = 4;
    constexpr size_t kBatchDim       = 2;
    constexpr size_t kMaxBlocksBatch = 4;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xCC);

    // batch 0: tokens=9, (9-1)%4==0 → boundary → zero block at idx 2
    //   block_ids: [1,2,5,0] → idx 2 → block_id 5 → ZEROED
    // batch 1: tokens=1, (1-1)%4==0 → boundary → zero block at idx 0
    //   block_ids: [3,0,0,0] → idx 0 → block_id 3 → ZEROED
    std::vector<int32_t> token_counts = {9, 1};
    std::vector<int32_t> block_ids    = {1,2,5,0, 3,0,0,0};

    auto* d_tc   = toDevice(token_counts);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, nullptr,
        2, kLayers, kBatchDim, kMaxBlocksBatch, kStride, kSeqPerBlock, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kLayers * kBlocks * kStride);
    for (size_t l = 0; l < kLayers; ++l) {
        const uint8_t* layer = result.data() + l * kBlocks * kStride;
        EXPECT_TRUE(isAllValue(layer + 5 * kStride, kStride, 0x00)) << "l=" << l << " block 5 zeroed";
        EXPECT_TRUE(isAllValue(layer + 3 * kStride, kStride, 0x00)) << "l=" << l << " block 3 zeroed";
        EXPECT_TRUE(isAllValue(layer + 1 * kStride, kStride, 0xCC)) << "l=" << l << " block 1 untouched";
        EXPECT_TRUE(isAllValue(layer + 2 * kStride, kStride, 0xCC)) << "l=" << l << " block 2 untouched";
    }

    cudaFree(d_tc);
    cudaFree(d_bids);
}

// Mid-block tokens are skipped by the kernel's modulo guard — no host-side filtering needed.
TEST_F(CudaBlockZeroKernelTest, SkipsMidBlockTokens) {
    constexpr size_t kLayers         = 1;
    constexpr size_t kBlocks         = 8;
    constexpr size_t kStride         = 256;
    constexpr size_t kSeqPerBlock    = 4;
    constexpr size_t kBatchDim       = 2;
    constexpr size_t kMaxBlocksBatch = 4;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xCC);

    // batch 0: tokens=10, (10-1)%4==1 → NOT boundary → skip
    // batch 1: tokens=3,  (3-1)%4==2  → NOT boundary → skip
    std::vector<int32_t> token_counts = {10, 3};
    std::vector<int32_t> block_ids    = {1,2,5,0, 3,0,0,0};

    auto* d_tc   = toDevice(token_counts);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, nullptr,
        2, kLayers, kBatchDim, kMaxBlocksBatch, kStride, kSeqPerBlock, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kBlocks * kStride);
    EXPECT_TRUE(isAllValue(result.data(), result.size(), 0xCC)) << "everything untouched";

    cudaFree(d_tc);
    cudaFree(d_bids);
}

// Multi-group: layer_to_group routes layers to different block_id groups.
TEST_F(CudaBlockZeroKernelTest, MultiGroupLayerMapping) {
    constexpr size_t kLayers         = 2;
    constexpr size_t kBlocks         = 8;
    constexpr size_t kStride         = 128;
    constexpr size_t kSeqPerBlock    = 4;
    constexpr size_t kBatchDim       = 1;
    constexpr size_t kMaxBlocksBatch = 4;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xBB);

    // tokens=5, (5-1)%4==0 → boundary → zero block at idx 1
    // layer 0 -> group 0: block_ids [1,3,0,0] -> idx 1 -> block_id 3
    // layer 1 -> group 1: block_ids [2,6,0,0] -> idx 1 -> block_id 6
    std::vector<int32_t> token_counts    = {5};
    std::vector<int32_t> layer_to_group  = {0, 1};
    std::vector<int32_t> block_ids       = {1,3,0,0, 2,6,0,0};

    auto* d_tc  = toDevice(token_counts);
    auto* d_ltg = toDevice(layer_to_group);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, d_ltg,
        1, kLayers, kBatchDim, kMaxBlocksBatch, kStride, kSeqPerBlock, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kLayers * kBlocks * kStride);
    EXPECT_TRUE(isAllValue(result.data() + 0 * kBlocks * kStride + 3 * kStride, kStride, 0x00))
        << "layer 0 block 3 zeroed (group 0)";
    EXPECT_TRUE(isAllValue(result.data() + 0 * kBlocks * kStride + 1 * kStride, kStride, 0xBB))
        << "layer 0 block 1 untouched";
    EXPECT_TRUE(isAllValue(result.data() + 1 * kBlocks * kStride + 6 * kStride, kStride, 0x00))
        << "layer 1 block 6 zeroed (group 1)";
    EXPECT_TRUE(isAllValue(result.data() + 1 * kBlocks * kStride + 2 * kStride, kStride, 0xBB))
        << "layer 1 block 2 untouched";

    cudaFree(d_tc);
    cudaFree(d_ltg);
    cudaFree(d_bids);
}

// Skips invalid block IDs (<=0) and zero token counts.
TEST_F(CudaBlockZeroKernelTest, SkipsInvalidBlocksAndZeroTokens) {
    constexpr size_t kLayers         = 1;
    constexpr size_t kBlocks         = 4;
    constexpr size_t kStride         = 64;
    constexpr size_t kBatchDim       = 2;
    constexpr size_t kMaxBlocksBatch = 2;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xFF);

    // batch 0: token_counts=0 -> skip (tokens <= 0)
    // batch 1: token_counts=1 -> (1-1)%4==0 boundary, but block_id=-1 -> skip (invalid)
    std::vector<int32_t> token_counts = {0, 1};
    std::vector<int32_t> block_ids    = {0,0, -1,0};

    auto* d_tc   = toDevice(token_counts);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, nullptr,
        2, kLayers, kBatchDim, kMaxBlocksBatch, kStride, 4, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kBlocks * kStride);
    EXPECT_TRUE(isAllValue(result.data(), result.size(), 0xFF)) << "everything untouched";

    cudaFree(d_tc);
    cudaFree(d_bids);
}

// Empty inputs: no crash, no-op.
TEST_F(CudaBlockZeroKernelTest, EmptyInputNoOp) {
    invokeZeroIncompleteKvCacheBlocks(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 256, 4, stream_);
    invokeZeroIncompleteKvCacheBlocks(nullptr, nullptr, nullptr, nullptr, 0, 4, 0, 0, 256, 4, stream_);
    invokeZeroIncompleteKvCacheBlocks(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 4, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
