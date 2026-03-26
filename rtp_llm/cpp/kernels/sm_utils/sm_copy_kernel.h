#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

// 支持N卡 & PPU
// COPY FROM tair mempool (pace)，就不改成RTP的namespace了，保留原namespace
// RTP只需要gather和scatter操作

namespace sDevMPS {
/**
 * @brief 启动一个 CUDA 内核，执行 Gather-Copy 操作。
 *
 * 该函数将多个源内存区域（每个由 src_ptrs[i] 指向）中相同偏移 offset 处、长度为 size 字节的数据，
 * 依次复制到连续的目标内存 dst 中。最终 dst 中的数据布局为：
 * [src_ptrs[0][offset:offset+size], src_ptrs[1][offset:offset+size], ..., src_ptrs[num_srcs-1][...]]
 *
 * 典型应用场景：从多个分散的缓冲区中提取相同位置的数据块，合并成一个连续缓冲区（如多个Layer的KVCache等）。
 *
 * @param src_ptrs      指向源内存指针的数组，长度为 num_srcs。每个元素是一个 void*，指向一个源缓冲区。
 *                      注意：这个指针数组，且里面每个元素指向的地址，均需SM可访问。
 * @param offset        每个源缓冲区中数据起始的统一偏移量（以字节为单位）。所有源都从 offset 处开始读取。
 * @param size          每个源要复制的数据大小（字节数）。所有源复制相同长度。
 * @param dst           目标内存的起始地址。必须已分配足够空间（num_srcs * size字节），该地址需SM可访问。
 * @param num_srcs      源缓冲区的数量，即 src_ptrs 数组的长度。必须 >= 0
 * @param block_num     Block数，用于控制使用的SM数，传入值为0时默认使用所有的SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 */
void launch_gather_copy(
    const void** src_ptrs, size_t offset, size_t size, void* dst, int num_srcs, int block_num, cudaStream_t stream);

/**
 * @brief 启动一个 CUDA 内核，执行 Scatter-Copy 操作。
 *
 * 该函数将连续源内存 src 中的数据，按块分割后分别复制到多个目标内存区域（由 dst_ptrs[i] 指向）中，
 * 每个目标从其 offset 偏移处开始写入 size 字节。数据布局为：
 * src[0:size] -> dst_ptrs[0][offset:offset+size]
 * src[size:2*size] -> dst_ptrs[1][offset:offset+size]
 * ...
 *
 * 典型应用场景：将一个大缓冲区的数据分发到多个独立缓冲区。
 *
 * @param src           源内存的起始地址。必须包含至少 num_dsts * size 字节的有效数据，该地址需SM可访问。
 * @param offset        每个目标缓冲区中写入起始的统一偏移量（字节）。所有目标都从 offset 处开始写入。
 * @param size          每个目标要写入的数据大小（字节）。所有目标写入相同长度。
 * @param dst_ptrs      指向目标内存指针的数组，长度为 num_dsts。每个元素是一个 void*，指向一个目标缓冲区。
 *                      注意：这个指针数组，且里面每个元素指向的地址，均需SM可访问。
 * @param num_dsts      目标缓冲区的数量，即 dst_ptrs 数组的长度。必须 >= 0。
 * @param block_num     Block数，用于控制使用的SM数，传入值为0时默认使用所有的SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 与 launch_gather_copy 互为逆操作（在相同参数下）。
 * @note 要求所有目标缓冲区在 [offset, offset + size) 范围内不重叠，否则行为未定义。
 */
void launch_scatter_copy(
    const void* src, size_t offset, size_t size, void** dst_ptrs, int num_dsts, int block_num, cudaStream_t stream);

/**
 * @brief Scatter from contiguous src to num_dsts pairs (dst_kv_cache[i], dst_kv_scale[i]).
 * Src layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...]; stride = kv_cache_size + kv_scale_size per dst.
 */
void launch_scatter_copy_split(const void*  src,
                               void**       dst_kv_cache_ptrs,
                               void**       dst_kv_scale_ptrs,
                               size_t       kv_cache_size,
                               size_t       kv_scale_size,
                               int          num_dsts,
                               int          block_num,
                               cudaStream_t stream);

/**
 * @brief Gather from num_srcs pairs (src_kv_cache[i], src_kv_scale[i]) to contiguous dst.
 * Dst layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...].
 */
void launch_gather_copy_split(const void** src_kv_cache_ptrs,
                              const void** src_kv_scale_ptrs,
                              size_t       kv_cache_size,
                              size_t       kv_scale_size,
                              void*        dst,
                              int          num_srcs,
                              int          block_num,
                              cudaStream_t stream);

/** JIT launch_scatter_copy_split / launch_gather_copy_split (benchmarks only; not used by noBlockCopyOpt). */
bool warmup_sm_copy_split_kernels(cudaStream_t stream);

/** JIT launch_scatter_copy_var_nooffset / launch_gather_copy_var_nooffset (CudaDevice noBlockCopyOpt path). */
bool warmup_sm_copy_var_nooffset_kernels(cudaStream_t stream);

/**
 * @brief 启动一个 CUDA 内核，执行变长 Gather-Copy 操作。
 *
 * 该函数将多个源内存区域（每个由 src_ptrs[i] 指向）中不同偏移和不同大小的数据，
 * 拷贝到连续的目标内存 dst 中。与 launch_gather_copy 不同，此函数支持每个源拥有独立的
 * 偏移量（src_offsets[i]）和拷贝大小（sizes[i]）。目标内存中的位置由 dst_offsets 前缀和数组确定。
 *
 * 数据流：
 *   src_ptrs[i] + src_offsets[i] --[sizes[i] bytes]--> dst + dst_offsets[i]
 *
 * 典型应用场景：KV Cache 场景中，不同层的 Page 大小可能不同（例如 K 和 V 的维度不同），
 * 需要将多个不连续的、大小各异的 GPU 内存区域 Gather 到 CPU 上的连续内存空间。
 *
 * @param src_ptrs      指向源内存指针的数组，长度为 num_srcs。每个元素是一个 void*，指向一个源缓冲区。
 *                      注意：这个指针数组，且里面每个元素指向的地址，均需 SM 可访问。
 * @param src_offsets   每个源的偏移量数组（设备端），长度为 num_srcs。src_offsets[i] 表示第 i 个源缓冲区
 *                      中数据的起始偏移（以字节为单位）。
 * @param sizes         每个源要拷贝的字节数数组（设备端），长度为 num_srcs。sizes[i] 表示从第 i 个源
 *                      拷贝的字节数。
 * @param dst_offsets   目标内存中的前缀和偏移量数组（设备端），长度为 num_srcs。dst_offsets[i] 表示
 *                      第 i 个源的数据在目标内存中的起始位置。通常由 Host 端预先计算：
 *                      dst_offsets[0] = 0, dst_offsets[i] = dst_offsets[i-1] + sizes[i-1] (i > 0)
 * @param dst           目标内存的起始地址。必须已分配足够空间（sum(sizes) 字节），该地址需 SM 可访问。
 * @param num_srcs      源缓冲区的数量，即 src_ptrs 数组的长度。必须 >= 0
 * @param block_num     Block 数，用于控制使用的 SM 数，传入值为 0 时默认使用所有的 SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 前缀和计算应在 Host 端完成，避免 Kernel 内部的全局同步开销。
 * @see launch_gather_copy 统一大小的 Gather 版本
 * @see launch_scatter_copy_var 变长 Scatter 版本
 */
void launch_gather_copy_var(const void**  src_ptrs,
                            const size_t* src_offsets,
                            const size_t* sizes,
                            const size_t* dst_offsets,
                            void*         dst,
                            int           num_srcs,
                            int           block_num,
                            cudaStream_t  stream);

/**
 * @brief 启动一个 CUDA 内核，执行变长 Scatter-Copy 操作。
 *
 * 该函数将连续源内存 src 中的数据，按不同偏移和不同大小分散拷贝到多个目标内存区域
 * （由 dst_ptrs[i] 指向）。与 launch_scatter_copy 不同，此函数支持每个目标拥有独立的
 * 偏移量（dst_offsets[i]）和拷贝大小（sizes[i]）。源内存中的位置由 src_offsets 前缀和数组确定。
 *
 * 数据流：
 *   src + src_offsets[i] --[sizes[i] bytes]--> dst_ptrs[i] + dst_offsets[i]
 *
 * 典型应用场景：KV Cache 场景中，将 CPU 上的连续内存空间 Scatter 到多个不连续的、大小各异
 * 的 GPU 内存区域，不同层的目标缓冲区可能需要不同大小的数据。
 *
 * @param src           源内存的起始地址。必须包含至少 sum(sizes) 字节的有效数据，该地址需 SM 可访问。
 * @param src_offsets   源内存中的前缀和偏移量数组（设备端），长度为 num_dsts。src_offsets[i] 表示
 *                      第 i 个目标的数据在源内存中的起始位置。通常由 Host 端预先计算：
 *                      src_offsets[0] = 0, src_offsets[i] = src_offsets[i-1] + sizes[i-1] (i > 0)
 * @param sizes         每个目标要拷贝的字节数数组（设备端），长度为 num_dsts。sizes[i] 表示拷贝到
 *                      第 i 个目标的字节数。
 * @param dst_offsets   每个目标的偏移量数组（设备端），长度为 num_dsts。dst_offsets[i] 表示第 i 个
 *                      目标缓冲区中写入的起始偏移（以字节为单位）。
 * @param dst_ptrs      指向目标内存指针的数组，长度为 num_dsts。每个元素是一个 void*，指向一个目标缓冲区。
 *                      注意：这个指针数组，且里面每个元素指向的地址，均需 SM 可访问。
 * @param num_dsts      目标缓冲区的数量，即 dst_ptrs 数组的长度。必须 >= 0。
 * @param block_num     Block 数，用于控制使用的 SM 数，传入值为 0 时默认使用所有的 SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 前缀和计算应在 Host 端完成，避免 Kernel 内部的全局同步开销。
 * @note 与 launch_gather_copy_var 互为逆操作（在相同参数下）。
 * @note 要求所有目标缓冲区在 [dst_offsets[i], dst_offsets[i] + sizes[i]) 范围内不重叠，否则行为未定义。
 * @see launch_scatter_copy 统一大小的 Scatter 版本
 * @see launch_gather_copy_var 变长 Gather 版本
 */
void launch_scatter_copy_var(const void*   src,
                             const size_t* src_offsets,
                             const size_t* sizes,
                             const size_t* dst_offsets,
                             void**        dst_ptrs,
                             int           num_dsts,
                             int           block_num,
                             cudaStream_t  stream);

/**
 * @brief 启动一个 CUDA 内核，执行优化版变长 Gather-Copy 操作（GPU 端指针已预偏移）。
 *
 * 该函数是 launch_gather_copy_var 的优化版本。调用方在 Host 端预先将 GPU 指针偏移到目标位置，
 * 从而让 Kernel 内部无需再做 src_offsets 加法运算。这样可以减少 Kernel 参数数量和内存访问，
 * 简化地址计算。
 *
 * 数据流：
 *   src_ptrs[i] --[sizes[i] bytes]--> dst + dst_offsets[i]
 *
 * 与 launch_gather_copy_var 的差异：
 *   - 移除了 src_offsets 参数，假设 src_ptrs[i] 已指向实际数据起始位置
 *   - 调用方需在 Host 端预计算偏移后的指针：device_ptrs_offset[i] = device_ptrs[i] + page_idx * page_sizes[i]
 *
 * 典型应用场景：当调用方可以在 Host 端方便地预计算偏移时，使用此优化版本可减少 Kernel 开销。
 *
 * @param src_ptrs      指向源内存指针的数组，长度为 num_srcs。每个元素是一个 void*，指向一个源缓冲区。
 *                      **注意**：每个指针应已预先偏移到实际数据的起始位置（即 src_ptrs[i] = base_ptr[i] + offset[i]）。
 *                      这个指针数组，且里面每个元素指向的地址，均需 SM 可访问。
 * @param sizes         每个源要拷贝的字节数数组（设备端），长度为 num_srcs。sizes[i] 表示从第 i 个源
 *                      拷贝的字节数。
 * @param dst_offsets   目标内存中的前缀和偏移量数组（设备端），长度为 num_srcs。dst_offsets[i] 表示
 *                      第 i 个源的数据在目标内存中的起始位置。通常由 Host 端预先计算：
 *                      dst_offsets[0] = 0, dst_offsets[i] = dst_offsets[i-1] + sizes[i-1] (i > 0)
 * @param dst           目标内存的起始地址。必须已分配足够空间（sum(sizes) 字节），该地址需 SM 可访问。
 * @param num_srcs      源缓冲区的数量，即 src_ptrs 数组的长度。必须 >= 0
 * @param block_num     Block 数，用于控制使用的 SM 数，传入值为 0 时默认使用所有的 SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 前缀和计算应在 Host 端完成，避免 Kernel 内部的全局同步开销。
 * @note 调用方需在 Host 端预计算偏移后的指针数组，并将偏移后的指针拷贝到 Device。
 * @see launch_gather_copy_var 带 offset 的变长 Gather 版本
 * @see launch_scatter_copy_var_nooffset 优化版变长 Scatter 版本
 */
void launch_gather_copy_var_nooffset(const void**  src_ptrs,
                                     const size_t* sizes,
                                     const size_t* dst_offsets,
                                     void*         dst,
                                     int           num_srcs,
                                     int           block_num,
                                     cudaStream_t  stream);

/**
 * @brief 启动一个 CUDA 内核，执行优化版变长 Scatter-Copy 操作（GPU 端指针已预偏移）。
 *
 * 该函数是 launch_scatter_copy_var 的优化版本。调用方在 Host 端预先将 GPU 指针偏移到目标位置，
 * 从而让 Kernel 内部无需再做 dst_offsets 加法运算。这样可以减少 Kernel 参数数量和内存访问，
 * 简化地址计算。
 *
 * 数据流：
 *   src + src_offsets[i] --[sizes[i] bytes]--> dst_ptrs[i]
 *
 * 与 launch_scatter_copy_var 的差异：
 *   - 移除了 dst_offsets 参数，假设 dst_ptrs[i] 已指向实际写入起始位置
 *   - 调用方需在 Host 端预计算偏移后的指针：device_ptrs_offset[i] = device_ptrs[i] + page_idx * page_sizes[i]
 *
 * 典型应用场景：当调用方可以在 Host 端方便地预计算偏移时，使用此优化版本可减少 Kernel 开销。
 *
 * @param src           源内存的起始地址。必须包含至少 sum(sizes) 字节的有效数据，该地址需 SM 可访问。
 * @param src_offsets   源内存中的前缀和偏移量数组（设备端），长度为 num_dsts。src_offsets[i] 表示
 *                      第 i 个目标的数据在源内存中的起始位置。通常由 Host 端预先计算：
 *                      src_offsets[0] = 0, src_offsets[i] = src_offsets[i-1] + sizes[i-1] (i > 0)
 * @param sizes         每个目标要拷贝的字节数数组（设备端），长度为 num_dsts。sizes[i] 表示拷贝到
 *                      第 i 个目标的字节数。
 * @param dst_ptrs      指向目标内存指针的数组，长度为 num_dsts。每个元素是一个 void*，指向一个目标缓冲区。
 *                      **注意**：每个指针应已预先偏移到实际写入的起始位置（即 dst_ptrs[i] = base_ptr[i] + offset[i]）。
 *                      这个指针数组，且里面每个元素指向的地址，均需 SM 可访问。
 * @param num_dsts      目标缓冲区的数量，即 dst_ptrs 数组的长度。必须 >= 0。
 * @param block_num     Block 数，用于控制使用的 SM 数，传入值为 0 时默认使用所有的 SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 前缀和计算应在 Host 端完成，避免 Kernel 内部的全局同步开销。
 * @note 调用方需在 Host 端预计算偏移后的指针数组，并将偏移后的指针拷贝到 Device。
 * @note 与 launch_gather_copy_var_nooffset 互为逆操作（在相同参数下）。
 * @see launch_scatter_copy_var 带 offset 的变长 Scatter 版本
 * @see launch_gather_copy_var_nooffset 优化版变长 Gather 版本
 */
void launch_scatter_copy_var_nooffset(const void*   src,
                                      const size_t* src_offsets,
                                      const size_t* sizes,
                                      void**        dst_ptrs,
                                      int           num_dsts,
                                      int           block_num,
                                      cudaStream_t  stream);

}  // namespace sDevMPS