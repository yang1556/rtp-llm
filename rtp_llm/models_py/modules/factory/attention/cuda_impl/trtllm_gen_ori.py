from typing import Optional

import flashinfer
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import is_sm_100
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
    KVCache,
    PyAttentionInputs,
)

# TRT-LLM kernels support at most 64 tokens per block (page_size constraint)
TRTLLM_MAX_PAGE_SIZE = 64


@triton.jit
def _kv_cache_offset_transform_kernel(
    ptr,
    block_num,
    stride_0,
    stride_1,
    stride_2,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: halve [:, 0, :] in-place and write [:, 1, :] = [:, 0, :] // 2 + block_num.

    Each program processes BLOCK_SIZE elements of the flattened (N, M) index space.
    Reads from dim-1=0 once, writes two outputs (dim-1=0 and dim-1=1) in a single pass,
    eliminating the intermediate read required by the sequential PyTorch version.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * M

    i = offsets // M
    j = offsets % M

    ptr_k = ptr + i * stride_0 + 0 * stride_1 + j * stride_2
    ptr_v = ptr + i * stride_0 + 1 * stride_1 + j * stride_2

    val = tl.load(ptr_k, mask=mask)
    # Arithmetic right shift is equivalent to floor-div by 2 for non-negative integers.
    val_half = val >> 1
    tl.store(ptr_k, val_half, mask=mask)
    tl.store(ptr_v, val_half + block_num, mask=mask)


def kv_cache_offset_transform(kv_cache_offset: torch.Tensor, block_num: int) -> None:
    """In-place fused transform on a 4-D integer offset tensor shaped [N, 1, 2, M].

    Equivalent to (but faster than) the two-step PyTorch sequence::

        kv_cache_offset[:, :, 0, :] //= 2
        kv_cache_offset[:, :, 1, :] = kv_cache_offset[:, 0, :] + block_num

    The Triton kernel fuses both writes into a single pass over the data,
    halving memory traffic compared with the naive approach.
    """
    kv_cache_offset = kv_cache_offset.squeeze(1)
    print("kkkk:", kv_cache_offset.shape, kv_cache_offset, flush=True)
    assert kv_cache_offset.dim() == 3 and kv_cache_offset.shape[1] == 2
    N, _, M = kv_cache_offset.shape
    BLOCK_SIZE = 512
    grid = ((N * M + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _kv_cache_offset_transform_kernel[grid](
        kv_cache_offset,
        block_num,
        kv_cache_offset.stride(0),
        kv_cache_offset.stride(1),
        kv_cache_offset.stride(2),
        N,
        M,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    print("kkkk2:", kv_cache_offset.shape, kv_cache_offset, flush=True)


def expand_block_tables_for_trtllm(
    block_tables: torch.Tensor,
    page_ratio: int,
) -> torch.Tensor:
    """Expand block_tables when splitting large blocks into small blocks.

    Each original block index b maps to contiguous small block indices
    [b*ratio, b*ratio+1, ..., b*ratio+ratio-1], matching the sub-block
    ordering produced by reshape_kv_cache_for_trtllm.

    Args:
        block_tables: original block tables, shape [batch_size, max_pages_per_seq]
        ratio: orig_page_size // trtllm_page_size

    Returns:
        Expanded block tables, shape [batch_size, max_pages_per_seq * page_ratio]
    """

    if page_ratio < 1:
        return block_tables
    B, max_pages = block_tables.shape
    print("block_tables1:", block_tables, flush=True)
    offsets = torch.arange(
        page_ratio, device=block_tables.device, dtype=block_tables.dtype
    )
    # [B, max_pages, ratio] -> [B, max_pages * ratio]

    block_tables = (
        (block_tables.unsqueeze(-1) * page_ratio + offsets)
        .view(B, max_pages * page_ratio)
        .cuda()
    )
    print("block_tables2:", block_tables, flush=True)
    return block_tables


# Constants
DEFAULT_TRT_WORKSPACE_SIZE_MB = (
    512  # Memory workspace size in MB, todo(Yingyi): read from config
)

# Global workspace buffer pool
_g_trt_workspace_pool: list[torch.Tensor] = []
_g_trt_pool_lock = __import__("threading").Lock()


def get_trt_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    """Get a TRT workspace buffer from the pool.

    This function manages a pool of workspace buffers to support multiple
    concurrent instances while avoiding excessive memory allocation.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda")

    Returns:
        Workspace buffer tensor of size DEFAULT_TRT_WORKSPACE_SIZE_MB
    """
    with _g_trt_pool_lock:
        if _g_trt_workspace_pool:
            return _g_trt_workspace_pool.pop()
        else:
            # No available buffer in pool, create a new one
            return torch.zeros(
                DEFAULT_TRT_WORKSPACE_SIZE_MB * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )


def release_trt_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a TRT workspace buffer back to the pool.

    Args:
        buffer: The workspace buffer to release
    """
    with _g_trt_pool_lock:
        _g_trt_workspace_pool.append(buffer)


class FlashInferTRTLLMParams(object):
    def __init__(
        self,
        batch_size: int,
        max_q_len: int = 0,
        max_kv_len: int = 0,
        max_seq_len: int = 0,
        seq_lens: Optional[torch.Tensor] = None,
        input_lens: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
    ):

        self.batch_size = batch_size
        self.max_q_len = max_q_len  # for prefill
        self.max_kv_len = max_kv_len  # for prefill
        self.max_seq_len = max_seq_len  # for decode
        self.seq_lens = seq_lens
        self.input_lens = input_lens
        self.block_tables = block_tables
        self.cu_seqlens = cu_seqlens
        self.cu_kv_seqlens = cu_kv_seqlens


class FlashInferTRTLLMPrefillOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
    ):
        print("init prefill:", flush=True)

        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.local_head_num = attn_configs.head_num
        self.local_head_kv_num = attn_configs.kv_head_num
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.workspace_buffer = get_trt_workspace_buffer()
        assert (
            self.seq_size_per_block <= TRTLLM_MAX_PAGE_SIZE
            or self.seq_size_per_block % TRTLLM_MAX_PAGE_SIZE == 0
        )
        self.page_size = min(self.seq_size_per_block, TRTLLM_MAX_PAGE_SIZE)
        self.page_ratio = self.seq_size_per_block // self.page_size

    def __del__(self):
        """Release workspace buffer back to pool when object is destroyed."""
        release_trt_workspace_buffer(self.workspace_buffer)

    def support(self, attention_inputs: PyAttentionInputs):
        print(
            "support prefill:",
            is_sm_100(),
            attention_inputs.is_prefill,
            attention_inputs.kv_cache_block_id_device is not None,
            flush=True,
        )
        return (
            is_sm_100()
            and attention_inputs.is_prefill
            and attention_inputs.kv_cache_block_id_device is not None
        )

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
        prefix_lengths = torch.zeros_like(
            attention_inputs.input_lengths,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        input_lengths = torch.zeros_like(
            attention_inputs.input_lengths,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        prefix_lengths.copy_(attention_inputs.prefix_lengths, non_blocking=True)
        input_lengths.copy_(attention_inputs.input_lengths, non_blocking=True)
        sequence_lengths = input_lengths + prefix_lengths
        page_per_seq = (sequence_lengths + self.page_size - 1) // self.page_size
        cu_kv_seqlens = torch.zeros(
            attention_inputs.input_lengths.shape[0] + 1,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        cu_kv_seqlens[1:] = torch.cumsum(page_per_seq, dim=0, dtype=torch.int32)

        print("bt:", attention_inputs.kv_cache_block_id_device, flush=True)
        return FlashInferTRTLLMParams(
            batch_size=attention_inputs.input_lengths.size(0),
            max_q_len=attention_inputs.input_lengths.max().item(),
            max_kv_len=(
                attention_inputs.prefix_lengths + attention_inputs.input_lengths
            )
            .max()
            .item(),
            seq_lens=sequence_lengths,
            input_lens=attention_inputs.input_lengths,
            block_tables=attention_inputs.kv_cache_block_id_device,
            cu_seqlens=attention_inputs.cu_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: FlashInferTRTLLMParams,
    ) -> torch.Tensor:
        dtype = kv_cache.kv_cache_base.dtype
        q_type = q.dtype
        q = q.to(dtype)
        o_type = q_type
        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)
        q_scale = 1.0
        k_scale = 1.0
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = 1.0
        if kv_cache:
            kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                2,
                kv_cache.kv_cache_base.shape[0] * self.page_ratio,
                self.local_head_kv_num,
                self.page_size,
                self.head_dim,
            )
        print("bt:", fmha_params.block_tables, flush=True)
        print("q:", q, flush=True)
        print(
            "k cache:",
            kv_cache.kv_cache_base[0].shape,
            kv_cache.kv_cache_base[0, 1, :, 0, :],
            flush=True,
        )
        print(
            "v cache:",
            kv_cache.kv_cache_base[1].shape,
            kv_cache.kv_cache_base[1, 1, :, 0, :],
            flush=True,
        )
        o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(kv_cache.kv_cache_base[0], kv_cache.kv_cache_base[1]),
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_q_len=fmha_params.max_q_len,
            max_kv_len=fmha_params.max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=fmha_params.batch_size,
            cum_seq_lens_q=fmha_params.cu_seqlens,
            cum_seq_lens_kv=fmha_params.cu_kv_seqlens,
            window_left=-1,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
            # kv_layout='NHD'
        )
        print("o:", o, flush=True)
        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)


class FlashInferTRTLLMDecodeOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
    ):
        print("init decode:", flush=True)
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.local_head_num = attn_configs.head_num
        self.local_head_kv_num = attn_configs.kv_head_num
        self.workspace_buffer = get_trt_workspace_buffer()
        assert (
            self.seq_size_per_block <= TRTLLM_MAX_PAGE_SIZE
            or self.seq_size_per_block % TRTLLM_MAX_PAGE_SIZE == 0
        )
        self.page_size = min(self.seq_size_per_block, TRTLLM_MAX_PAGE_SIZE)
        self.page_ratio = self.seq_size_per_block // self.page_size

    def __del__(self):
        """Release workspace buffer back to pool when object is destroyed."""
        release_trt_workspace_buffer(self.workspace_buffer)

    def support(self, attention_inputs: PyAttentionInputs):
        print(
            "support decode:",
            is_sm_100(),
            attention_inputs.is_prefill,
            attention_inputs.input_lengths,
            flush=True,
        )
        if not is_sm_100():
            return False
        if (
            attention_inputs.is_prefill
            and attention_inputs.input_lengths[0] < 10
            and (attention_inputs.input_lengths == attention_inputs.input_lengths[0])
            .all()
            .item()
        ):
            return True
        return not attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
        if not attention_inputs.is_prefill:
            # need transfer to cuda, cuda graph can capture the add
            sequence_lengths = torch.ones_like(
                attention_inputs.sequence_lengths,
                device="cuda",
                dtype=attention_inputs.sequence_lengths.dtype,
            )
            sequence_lengths.copy_(
                attention_inputs.sequence_lengths, non_blocking=True
            ).add_(1)

            return FlashInferTRTLLMParams(
                batch_size=attention_inputs.sequence_lengths.size(0),
                max_seq_len=attention_inputs.sequence_lengths.max().item() + 1,
                seq_lens=sequence_lengths,
                block_tables=attention_inputs.kv_cache_block_id_device,
            )
        else:
            q_len = attention_inputs.input_lengths[0].item()
            sequence_lengths = torch.zeros_like(
                attention_inputs.prefix_lengths,
                device="cuda",
                dtype=attention_inputs.prefix_lengths.dtype,
            )
            sequence_lengths.copy_(
                attention_inputs.prefix_lengths, non_blocking=True
            ).add_(q_len)
            return FlashInferTRTLLMParams(
                batch_size=attention_inputs.prefix_lengths.size(0),
                max_seq_len=attention_inputs.prefix_lengths.max().item() + q_len,
                seq_lens=sequence_lengths,
                block_tables=attention_inputs.kv_cache_block_id_device,
            )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: FlashInferTRTLLMParams,
    ) -> torch.Tensor:
        dtype = kv_cache.kv_cache_base.dtype
        q_type = q.dtype
        q = q.to(dtype)
        o_type = q_type

        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)
        q_scale = 1.0
        k_scale = 1.0
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = 1.0
        # sink: additional value per head in the denominator of the softmax.
        if kv_cache:
            kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                2,
                kv_cache.kv_cache_base.shape[0] * self.page_ratio,
                self.local_head_kv_num,
                self.page_size,
                self.head_dim,
            )

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        print("dbt:", fmha_params.block_tables, flush=True)
        print("dq:", q, flush=True)
        print(
            "dk cache:",
            kv_cache.kv_cache_base[0].shape,
            kv_cache.kv_cache_base[0, 1, :, 0, :],
            flush=True,
        )
        print(
            "dv cache:",
            kv_cache.kv_cache_base[1].shape,
            kv_cache.kv_cache_base[1, 1, :, 0, :],
            flush=True,
        )
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(kv_cache.kv_cache_base[0], kv_cache.kv_cache_base[1]),
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_seq_len=fmha_params.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=-1,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
            # kv_layout="NHD",
            q_len_per_req=q.shape[0] // fmha_params.seq_lens.shape[0],
        )
        print("do:", o, flush=True)

        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)


class FlashInferTRTLLMPrefillImplOri(FMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMPrefillOp(attn_configs)
        if attn_configs.tokens_per_block > TRTLLM_MAX_PAGE_SIZE:
            attn_configs.tokens_per_block = TRTLLM_MAX_PAGE_SIZE
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.kv_cache_block_id_device_ori = attn_inputs.kv_cache_block_id_device
        attn_inputs.kv_cache_block_id_device = expand_block_tables_for_trtllm(
            attn_inputs.kv_cache_block_id_device, self.fmha_impl.page_ratio
        )
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        attn_inputs.kv_cache_block_id_device = self.kv_cache_block_id_device_ori

        self.used = True

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Create temporary instance to check support
        fmha_impl = FlashInferTRTLLMPrefillOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.used:
                kv_cache_offset_transform(
                    self.rope_params.kv_cache_offset,
                    kv_cache.kv_cache_base.shape[0] * self.fmha_impl.page_ratio,
                )
                self.used = False
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_params.seq_lens.copy_(new_fmha_params.seq_lens, non_blocking=True)
        self.fmha_params.cu_kv_seqlens.copy_(
            new_fmha_params.cu_kv_seqlens, non_blocking=True
        )

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        common.copy_kv_cache_offset(old_offset, new_offset)


class FlashInferTRTLLMSpecDecodeImplOri(FMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        if attn_configs.tokens_per_block > TRTLLM_MAX_PAGE_SIZE:
            attn_configs.tokens_per_block = TRTLLM_MAX_PAGE_SIZE
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.kv_cache_block_id_device_ori = attn_inputs.kv_cache_block_id_device
        attn_inputs.kv_cache_block_id_device = expand_block_tables_for_trtllm(
            attn_inputs.kv_cache_block_id_device, self.fmha_impl.page_ratio
        )
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        attn_inputs.kv_cache_block_id_device = self.kv_cache_block_id_device_ori

        self.used = True

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Check MLA is not enabled
        if attn_configs.use_mla:
            return False
        # Create temporary instance to check support
        fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.used:
                kv_cache_offset_transform(
                    self.rope_params.kv_cache_offset,
                    kv_cache.kv_cache_base.shape[0] * self.fmha_impl.page_ratio,
                )
                self.used = False
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_params.seq_lens.copy_(new_fmha_params.seq_lens, non_blocking=True)

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        common.copy_kv_cache_offset(old_offset, new_offset)


class FlashInferTRTLLMDecodeImplOri(FMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        if attn_configs.tokens_per_block > TRTLLM_MAX_PAGE_SIZE:
            attn_configs.tokens_per_block = TRTLLM_MAX_PAGE_SIZE
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params

        self.kv_cache_block_id_device_ori = attn_inputs.kv_cache_block_id_device
        attn_inputs.kv_cache_block_id_device = expand_block_tables_for_trtllm(
            attn_inputs.kv_cache_block_id_device, self.fmha_impl.page_ratio
        )
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        attn_inputs.kv_cache_block_id_device = self.kv_cache_block_id_device_ori
        self.used = True

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Check MLA is not enabled
        if attn_configs.use_mla:
            return False
        # Create temporary instance to check support
        fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.used:
                kv_cache_offset_transform(
                    self.rope_params.kv_cache_offset,
                    kv_cache.kv_cache_base.shape[0] * self.fmha_impl.page_ratio,
                )
                self.used = False
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_params.seq_lens.copy_(new_fmha_params.seq_lens, non_blocking=True)

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        common.copy_kv_cache_offset(old_offset, new_offset)
