import math
import os
import random
import sys
from typing import List, Optional
from unittest import SkipTest, TestCase, main

import torch
import triton

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.atten_test_util import (
    attention_prefill_ref,
    gen_attention_inputs,
    write_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    FlashInferTRTLLMDecodeOp,
    FlashInferTRTLLMPrefillOp,
    _prepare_cuda_graph_kernel,
)
from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance

device = torch.device("cuda")

from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlashInferPythonMHATest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda")
        set_seed(25536)
        self.num_pages = 1024
        self.page_size = 64
        self.head_dim = 128
        self.num_kv_heads = 8
        self.num_heads = 64

    def _init_kv_cache(self, dtype: torch.dtype = torch.float8_e4m3fn) -> LayerKVCache:
        k_cache = (
            torch.rand(
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            * 2
            - 1
        ).to(dtype)
        v_cache = (
            torch.rand(
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            * 2
            - 1
        ).to(dtype)
        kv_cache: LayerKVCache = LayerKVCache()
        kv_cache.kv_cache_base = torch.stack([k_cache, v_cache], dim=1)
        return kv_cache

    def _create_config(self, dtype) -> AttentionConfigs:
        """Create a standard AttentionConfigs config for testing."""
        config = AttentionConfigs()
        config.head_num = self.num_heads
        config.kv_head_num = self.num_kv_heads
        config.size_per_head = self.head_dim
        config.tokens_per_block = self.page_size
        config.kernel_tokens_per_block = self.page_size
        config.kv_cache_dtype = (
            KvCacheDataType.FP8
            if dtype is torch.float8_e4m3fn
            else KvCacheDataType.BASE
        )
        config.use_mla = False
        config.is_causal = True
        config.fuse_qkv_add_bias = True
        config.q_scaling = 1.0
        return config

    def _test_flashinfer_trtllm_prefill_base(self, dtype: torch.dtype):
        """Test FlashInferTRTLLM prefill attention with reference comparison."""
        # Check if SM_100 is available
        is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
        if not is_sm_100:
            raise SkipTest("FlashInferTRTLLM requires SM_100 (compute capability 10.0)")
        input_lengths = [2, 129, 255, 63]
        num_tokens = sum(input_lengths)
        config = self._create_config(dtype)
        attn_inputs = gen_attention_inputs(
            self.page_size, self.num_pages, input_lengths=input_lengths
        )
        hidden_size = self.head_dim * self.num_heads
        qkv = (
            torch.rand(
                [
                    num_tokens,
                    hidden_size + 2 * self.num_kv_heads * self.head_dim,
                ],
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 2
            - 1
        )

        # Split qkv for reference implementation
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads
        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size : q_size + k_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v_ref = qkv[:, q_size + k_size : q_size + k_size + v_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        kv_cache = self._init_kv_cache(dtype)
        write_kv_cache(
            k_ref,
            v_ref,
            kv_cache,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
        )
        # Run FlashInferTRTLLM implementation
        op = FlashInferTRTLLMPrefillOp(config)
        input_params = op.prepare(attn_inputs)
        out_trtllm = op.forward(q_ref, kv_cache, input_params)
        # Run reference implementation
        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.sequence_lengths,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        # Reshape output to match reference
        out_trtllm_reshaped = out_trtllm.reshape(
            num_tokens, self.num_heads, self.head_dim
        )
        # Convert to float32 for comparison
        out_trtllm_f32 = out_trtllm_reshaped.float()
        out_ref_f32 = out_ref.float()
        atol = 0.04
        rtol = 0.04
        allowed_mismatch_rate = 1e-5
        assert_close_with_mismatch_tolerance(
            out_trtllm_f32,
            out_ref_f32,
            atol=atol,
            rtol=rtol,
            max_mismatched_elements=int(allowed_mismatch_rate * out_ref_f32.numel()),
        )

    def _test_flashinfer_trtllm_decode_base(self, dtype: torch.dtype):
        """Test FlashInferTRTLLM decode attention with reference comparison."""
        # Check if SM_100 is available
        is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
        if not is_sm_100:
            raise SkipTest("FlashInferTRTLLM requires SM_100 (compute capability 10.0)")
        sequence_lengths = [2, 129, 255, 63]
        batch_size = len(sequence_lengths)
        num_tokens = sum(sequence_lengths)
        config = self._create_config(dtype)
        attn_inputs = gen_attention_inputs(
            self.page_size, self.num_pages, sequence_lengths=sequence_lengths
        )
        hidden_size = self.head_dim * self.num_heads
        qkv = (
            torch.rand(
                [
                    num_tokens,
                    hidden_size + 2 * self.num_kv_heads * self.head_dim,
                ],
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 2
            - 1
        )
        # Split qkv for reference implementation
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads
        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size : q_size + k_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v_ref = qkv[:, q_size + k_size : q_size + k_size + v_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        last_token_idx = attn_inputs.cu_seqlens[1:] - 1
        # Run reference implementation
        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.sequence_lengths,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        out_ref = out_ref[last_token_idx]
        kv_cache = self._init_kv_cache(dtype)
        write_kv_cache(
            k_ref,
            v_ref,
            kv_cache,
            attn_inputs.sequence_lengths,
            attn_inputs.kv_cache_block_id_host,
        )

        op = FlashInferTRTLLMDecodeOp(config)
        q = q_ref[last_token_idx]
        attn_inputs.sequence_lengths -= 1
        input_params = op.prepare(attn_inputs)
        out_trtllm = op.forward(q, kv_cache, input_params)
        # Reshape output to match reference
        out_trtllm_reshaped = out_trtllm.reshape(-1, self.num_heads, self.head_dim)
        out_trtllm_f32 = out_trtllm_reshaped.float()
        out_ref_f32 = out_ref.float()
        atol = 0.04  # More relaxed for float8
        rtol = 0.04
        allowed_mismatch_rate = 1e-3
        assert_close_with_mismatch_tolerance(
            out_trtllm_f32,
            out_ref_f32,
            atol=atol,
            rtol=rtol,
            max_mismatched_elements=int(allowed_mismatch_rate * out_ref_f32.numel()),
        )

    def test_flashinfer_trtllm_prefill_bf16(self):
        self._test_flashinfer_trtllm_prefill_base(torch.bfloat16)

    def test_flashinfer_trtllm_prefill_fp8(self):
        self._test_flashinfer_trtllm_prefill_base(torch.float8_e4m3fn)

    def test_flashinfer_trtllm_decode_bf16(self):
        self._test_flashinfer_trtllm_decode_base(torch.bfloat16)

    def test_flashinfer_trtllm_prefill_fp8(self):
        self._test_flashinfer_trtllm_decode_base(torch.float8_e4m3fn)


class PrepareCudaGraphKernelTest(TestCase):
    """Tests for the unified _prepare_cuda_graph_kernel Triton kernel."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda")
        set_seed(42)

    def _run_kernel(self, src1, src2, seq_lens_out, cu_kv_seqlens_out,
                    block_id, kv_offset_out, page_size, N, M, mode):
        total_bm = N * M
        BLOCK_SIZE = max(triton.next_power_of_2(N), 1024)
        grid = (triton.cdiv(total_bm, BLOCK_SIZE),)
        _prepare_cuda_graph_kernel[grid](
            src1, src2,
            seq_lens_out, cu_kv_seqlens_out,
            block_id, kv_offset_out,
            page_size, N, M, total_bm,
            MODE=mode, BLOCK_SIZE=BLOCK_SIZE,
        )
        torch.cuda.synchronize()

    def _make_block_id(self, N, M):
        return torch.randint(0, 512, (N, M), dtype=torch.int32, device=self.device)

    def _reference_kv_offset(self, block_id):
        """Reference: block_id[B,M] -> kv_offset[B,2,M] with K=id*2, V=id*2+1."""
        B, M = block_id.shape
        kv_offset = torch.zeros(B, 2, M, dtype=torch.int32, device=self.device)
        for b in range(B):
            for m in range(M):
                bid = block_id[b, m].item()
                kv_offset[b, 0, m] = bid * 2
                kv_offset[b, 1, m] = bid * 2 + 1
        return kv_offset

    # ── MODE=0 (decode): seq_lens = copy(src1) ──

    def test_mode0_seq_lens_copy(self):
        N, M = 4, 8
        src1 = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=self.device)
        src2 = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=0)

        torch.testing.assert_close(seq_lens_out, src1)

    def test_mode0_kv_offset(self):
        N, M = 4, 8
        src1 = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=self.device)
        src2 = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=0)

        expected_kv = self._reference_kv_offset(block_id)
        torch.testing.assert_close(kv_offset, expected_kv)

    def test_mode0_large_batch(self):
        N, M = 128, 16
        src1 = torch.randint(1, 1000, (N,), dtype=torch.int32, device=self.device)
        src2 = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=0)

        torch.testing.assert_close(seq_lens_out, src1)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    # ── MODE=1 (spec-decode prefill): seq_lens = prefix + src2[0] ──

    def test_mode1_seq_lens_add_scalar(self):
        N, M = 4, 8
        prefix = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)
        q_len_tensor = torch.tensor([5], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(prefix, q_len_tensor, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=1)

        expected = prefix + 5
        torch.testing.assert_close(seq_lens_out, expected)

    def test_mode1_kv_offset(self):
        N, M = 4, 8
        prefix = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)
        q_len_tensor = torch.tensor([5], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(prefix, q_len_tensor, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=1)

        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_mode1_large_batch(self):
        N, M = 64, 32
        prefix = torch.randint(50, 500, (N,), dtype=torch.int32, device=self.device)
        q_len_tensor = torch.tensor([7], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(prefix, q_len_tensor, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=1)

        torch.testing.assert_close(seq_lens_out, prefix + 7)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    # ── MODE=2 (prefill): seq_lens = input + prefix, cu_kv_seqlens ──

    def test_mode2_seq_lens(self):
        N, M = 4, 8
        page_size = 64
        input_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=self.device)
        prefix_lens = torch.tensor([5, 15, 25, 35], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(input_lens, prefix_lens, seq_lens_out, cu_kv, block_id, kv_offset, page_size, N, M, mode=2)

        expected_seq = input_lens + prefix_lens
        torch.testing.assert_close(seq_lens_out, expected_seq)

    def test_mode2_cu_kv_seqlens(self):
        N, M = 4, 16
        page_size = 64
        input_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=self.device)
        prefix_lens = torch.tensor([5, 15, 25, 35], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(input_lens, prefix_lens, seq_lens_out, cu_kv, block_id, kv_offset, page_size, N, M, mode=2)

        total_seq = (input_lens + prefix_lens).cpu()
        pages_per_seq = (total_seq + page_size - 1) // page_size
        expected_cu = torch.zeros(N + 1, dtype=torch.int32)
        expected_cu[1:] = torch.cumsum(pages_per_seq, dim=0)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)

    def test_mode2_kv_offset(self):
        N, M = 4, 8
        page_size = 64
        input_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=self.device)
        prefix_lens = torch.tensor([5, 15, 25, 35], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(input_lens, prefix_lens, seq_lens_out, cu_kv, block_id, kv_offset, page_size, N, M, mode=2)

        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_mode2_large_batch(self):
        N, M = 128, 32
        page_size = 128
        input_lens = torch.randint(1, 500, (N,), dtype=torch.int32, device=self.device)
        prefix_lens = torch.randint(0, 200, (N,), dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(input_lens, prefix_lens, seq_lens_out, cu_kv, block_id, kv_offset, page_size, N, M, mode=2)

        expected_seq = input_lens + prefix_lens
        torch.testing.assert_close(seq_lens_out, expected_seq)

        total_cpu = expected_seq.cpu()
        pages = (total_cpu + page_size - 1) // page_size
        expected_cu = torch.zeros(N + 1, dtype=torch.int32)
        expected_cu[1:] = torch.cumsum(pages, dim=0)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    # ── Edge cases ──

    def test_single_batch(self):
        """Single-element batch for all modes."""
        M = 4
        for mode in [0, 1, 2]:
            src1 = torch.tensor([42], dtype=torch.int32, device=self.device)
            src2 = torch.tensor([10], dtype=torch.int32, device=self.device)
            seq_lens_out = torch.zeros(1, dtype=torch.int32, device=self.device)
            cu_kv = torch.zeros(2, dtype=torch.int32, device=self.device)
            block_id = self._make_block_id(1, M)
            kv_offset = torch.zeros(1, 2, M, dtype=torch.int32, device=self.device)
            page_size = 64 if mode == 2 else 0

            self._run_kernel(src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, page_size, 1, M, mode=mode)

            if mode == 0:
                self.assertEqual(seq_lens_out.item(), 42)
            elif mode == 1:
                self.assertEqual(seq_lens_out.item(), 42 + 10)
            else:
                self.assertEqual(seq_lens_out.item(), 42 + 10)
                expected_pages = (52 + 64 - 1) // 64
                self.assertEqual(cu_kv[0].item(), 0)
                self.assertEqual(cu_kv[1].item(), expected_pages)

            torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_mode2_page_boundary(self):
        """Sequences exactly on page boundaries."""
        N, M = 3, 4
        page_size = 64
        input_lens = torch.tensor([64, 128, 192], dtype=torch.int32, device=self.device)
        prefix_lens = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(input_lens, prefix_lens, seq_lens_out, cu_kv, block_id, kv_offset, page_size, N, M, mode=2)

        torch.testing.assert_close(seq_lens_out, input_lens)
        # Exact page boundaries: 1, 2, 3 pages
        expected_cu = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)

    def test_mode2_one_over_page_boundary(self):
        """Sequences one token over page boundaries need an extra page."""
        N, M = 3, 8
        page_size = 64
        input_lens = torch.tensor([65, 129, 193], dtype=torch.int32, device=self.device)
        prefix_lens = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(input_lens, prefix_lens, seq_lens_out, cu_kv, block_id, kv_offset, page_size, N, M, mode=2)

        # 65 -> 2 pages, 129 -> 3 pages, 193 -> 4 pages
        expected_cu = torch.tensor([0, 2, 5, 9], dtype=torch.int32)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)

    def test_large_M_multi_block(self):
        """total_bm exceeds BLOCK_SIZE, exercising multi-block grid."""
        N, M = 8, 256
        src1 = torch.randint(1, 100, (N,), dtype=torch.int32, device=self.device)
        src2 = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=0)

        torch.testing.assert_close(seq_lens_out, src1)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))


class FlashInferTRTLLMSpecDecodeImplTest(TestCase):
    """Tests for FlashInferTRTLLMSpecDecodeImpl.prepare_cuda_graph patterns.

    SpecDecodeImpl.prepare_cuda_graph has two branches:
      - Decode (is_prefill=False): MODE=0, src1=src2=sequence_lengths_plus_1_d,
        seq_lens_out=cu_kv_out=fmha_params.seq_lens (aliased), page_size=0
      - Prefill (is_prefill=True): MODE=1, src1=prefix_lengths_d, src2=input_lengths_d,
        seq_lens_out=cu_kv_out=fmha_params.seq_lens (aliased), page_size=0
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda")
        set_seed(42)

    def _make_block_id(self, N, M):
        return torch.randint(0, 512, (N, M), dtype=torch.int32, device=self.device)

    def _reference_kv_offset(self, block_id):
        B, M = block_id.shape
        kv_offset = torch.zeros(B, 2, M, dtype=torch.int32, device=self.device)
        for b in range(B):
            for m in range(M):
                bid = block_id[b, m].item()
                kv_offset[b, 0, m] = bid * 2
                kv_offset[b, 1, m] = bid * 2 + 1
        return kv_offset

    def _run_spec_decode_prepare(self, is_prefill, seq_lens_plus_1=None,
                                  prefix_lengths=None, input_lengths=None,
                                  block_id=None, kv_offset=None):
        """Simulate FlashInferTRTLLMSpecDecodeImpl.prepare_cuda_graph exactly."""
        N = block_id.shape[0]
        M = block_id.shape[1]
        total_bm = N * M
        BLOCK_SIZE = max(triton.next_power_of_2(N), 1024)
        grid = (triton.cdiv(total_bm, BLOCK_SIZE),)
        seq_lens = torch.zeros(N, dtype=torch.int32, device=self.device)
        if not is_prefill:
            _prepare_cuda_graph_kernel[grid](
                seq_lens_plus_1, seq_lens_plus_1,
                seq_lens, seq_lens,
                block_id, kv_offset,
                0, N, M, total_bm,
                MODE=0, BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _prepare_cuda_graph_kernel[grid](
                prefix_lengths, input_lengths,
                seq_lens, seq_lens,
                block_id, kv_offset,
                0, N, M, total_bm,
                MODE=1, BLOCK_SIZE=BLOCK_SIZE,
            )
        torch.cuda.synchronize()
        return seq_lens

    # ── Decode path (is_prefill=False) ──

    def test_decode_copies_seq_lens(self):
        """Decode: seq_lens = copy(sequence_lengths_plus_1)."""
        N, M = 4, 8
        seq_lens_plus_1 = torch.tensor([11, 21, 31, 41], dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=False, seq_lens_plus_1=seq_lens_plus_1,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, seq_lens_plus_1)

    def test_decode_kv_offset(self):
        """Decode: kv_offset correctly computed."""
        N, M = 4, 8
        seq_lens_plus_1 = torch.tensor([11, 21, 31, 41], dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_spec_decode_prepare(
            is_prefill=False, seq_lens_plus_1=seq_lens_plus_1,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_decode_aliased_src_tensors(self):
        """Decode: same tensor used as both src1 and src2 must not corrupt."""
        N, M = 8, 16
        seq_lens_plus_1 = torch.randint(1, 500, (N,), dtype=torch.int32, device=self.device)
        original = seq_lens_plus_1.clone()
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=False, seq_lens_plus_1=seq_lens_plus_1,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, original)
        torch.testing.assert_close(seq_lens_plus_1, original)

    def test_decode_large_batch(self):
        N, M = 128, 32
        seq_lens_plus_1 = torch.randint(2, 1000, (N,), dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=False, seq_lens_plus_1=seq_lens_plus_1,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, seq_lens_plus_1)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    # ── Prefill path (is_prefill=True) ──

    def test_prefill_seq_lens_prefix_plus_qlen(self):
        """Prefill: seq_lens = prefix_lengths + input_lengths[0] (broadcast scalar)."""
        N, M = 4, 8
        prefix = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)
        q_len = 5
        input_lengths = torch.full((N,), q_len, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, prefix + q_len)

    def test_prefill_kv_offset(self):
        """Prefill: kv_offset correctly computed."""
        N, M = 4, 8
        prefix = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)
        input_lengths = torch.full((N,), 5, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_prefill_varying_prefix_uniform_qlen(self):
        """Prefill: varying prefix lengths with uniform q_len (MTP/spec-decode pattern)."""
        N, M = 6, 16
        prefix = torch.tensor([50, 100, 150, 200, 250, 300], dtype=torch.int32, device=self.device)
        q_len = 3
        input_lengths = torch.full((N,), q_len, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, prefix + q_len)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_prefill_large_batch(self):
        N, M = 64, 32
        prefix = torch.randint(50, 500, (N,), dtype=torch.int32, device=self.device)
        q_len = 7
        input_lengths = torch.full((N,), q_len, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, prefix + q_len)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_prefill_qlen_1(self):
        """Prefill with q_len=1 (single-token speculative step)."""
        N, M = 4, 8
        prefix = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)
        input_lengths = torch.full((N,), 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, prefix + 1)

    # ── Mode switching ──

    def test_decode_then_prefill_reuses_output(self):
        """Switching from decode to prefill overwrites seq_lens correctly."""
        N, M = 4, 8
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens_plus_1 = torch.tensor([11, 21, 31, 41], dtype=torch.int32, device=self.device)
        seq_lens = self._run_spec_decode_prepare(
            is_prefill=False, seq_lens_plus_1=seq_lens_plus_1,
            block_id=block_id, kv_offset=kv_offset,
        )
        torch.testing.assert_close(seq_lens, seq_lens_plus_1)

        prefix = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)
        input_lengths = torch.full((N,), 5, dtype=torch.int32, device=self.device)
        kv_offset2 = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)
        block_id2 = self._make_block_id(N, M)

        seq_lens2 = self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id2, kv_offset=kv_offset2,
        )

        torch.testing.assert_close(seq_lens2, prefix + 5)
        torch.testing.assert_close(kv_offset2, self._reference_kv_offset(block_id2))

    def test_single_batch_decode(self):
        N, M = 1, 4
        seq_lens_plus_1 = torch.tensor([42], dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=False, seq_lens_plus_1=seq_lens_plus_1,
            block_id=block_id, kv_offset=kv_offset,
        )

        self.assertEqual(seq_lens.item(), 42)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_single_batch_prefill(self):
        N, M = 1, 4
        prefix = torch.tensor([100], dtype=torch.int32, device=self.device)
        input_lengths = torch.tensor([3], dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id, kv_offset=kv_offset,
        )

        self.assertEqual(seq_lens.item(), 103)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_multi_block_grid_decode(self):
        """total_bm exceeds BLOCK_SIZE, exercising multi-block grid in decode mode."""
        N, M = 4, 512
        seq_lens_plus_1 = torch.tensor([11, 21, 31, 41], dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=False, seq_lens_plus_1=seq_lens_plus_1,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, seq_lens_plus_1)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_multi_block_grid_prefill(self):
        """total_bm exceeds BLOCK_SIZE, exercising multi-block grid in prefill mode."""
        N, M = 4, 512
        prefix = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)
        input_lengths = torch.full((N,), 5, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        seq_lens = self._run_spec_decode_prepare(
            is_prefill=True, prefix_lengths=prefix, input_lengths=input_lengths,
            block_id=block_id, kv_offset=kv_offset,
        )

        torch.testing.assert_close(seq_lens, prefix + 5)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))


if __name__ == "__main__":
    main()
