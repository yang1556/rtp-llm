#pragma once

namespace rtp_llm {

/// Warm up split KV scatter/gather CUDA kernels on every device visible to this process (CUDA_VISIBLE_DEVICES).
/// Call once before `DeviceFactory::initDevices` / `CudaDevice` / NCCL (`EngineBase` ctor, `EmbeddingEngine` ctor).
void warmup_sm_copy_split_kernels_visible_cuda_devices();

}  // namespace rtp_llm
