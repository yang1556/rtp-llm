import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct


class TestCollectiveTorchHipGraphUnit(unittest.TestCase):
    def setUp(self):
        self._orig_is_rocm_runtime = ct._is_rocm_runtime
        self._orig_rccl_comm = ct._rccl_comm
        self._orig_rccl_world_size = ct._rccl_world_size
        self._orig_rccl_lib = ct._rccl_lib
        self._orig_cache = dict(ct._hipgraph_allgather_outputs)
        ct._hipgraph_allgather_outputs.clear()

    def tearDown(self):
        ct._is_rocm_runtime = self._orig_is_rocm_runtime
        ct._rccl_comm = self._orig_rccl_comm
        ct._rccl_world_size = self._orig_rccl_world_size
        ct._rccl_lib = self._orig_rccl_lib
        ct._hipgraph_allgather_outputs.clear()
        ct._hipgraph_allgather_outputs.update(self._orig_cache)

    def test_should_use_hipgraph_capture_rccl(self):
        ct._is_rocm_runtime = True
        ct._rccl_comm = ctypes.c_void_p(123)
        with patch.object(ct, "_is_hipgraph_capture_active", return_value=True):
            self.assertTrue(ct._should_use_hipgraph_capture_rccl(ct.Group.TP))
            self.assertFalse(ct._should_use_hipgraph_capture_rccl(ct.Group.DP))
            self.assertFalse(ct._should_use_hipgraph_capture_rccl(ct.Group.DP_AND_TP))

    def test_should_not_use_hipgraph_capture_rccl_without_comm(self):
        ct._is_rocm_runtime = True
        ct._rccl_comm = None
        with patch.object(ct, "_is_hipgraph_capture_active", return_value=True):
            self.assertFalse(ct._should_use_hipgraph_capture_rccl(ct.Group.TP))

    def test_get_nccl_dtype_map_and_error(self):
        fp16_tensor = torch.zeros(1, dtype=torch.float16)
        self.assertEqual(
            ct._get_nccl_dtype(fp16_tensor), ct._NCCL_DTYPE_MAP[torch.float16]
        )

        with self.assertRaises(TypeError):
            ct._get_nccl_dtype(torch.zeros(1, dtype=torch.bool))

    def test_get_or_create_allgather_output_cache_reuse(self):
        ct._rccl_world_size = 3
        src = torch.zeros((2, 4), dtype=torch.float16)

        with patch.object(ct, "_is_hipgraph_capture_active", return_value=True):
            out1 = ct._get_or_create_allgather_output(src)
            out2 = ct._get_or_create_allgather_output(src)

        self.assertIs(out1, out2)
        self.assertEqual(tuple(out1.shape), (6, 4))
        self.assertEqual(out1.dtype, src.dtype)
        self.assertEqual(out1.device, src.device)
        self.assertEqual(len(ct._hipgraph_allgather_outputs), 1)

    def test_get_or_create_allgather_output_rejects_inactive_capture(self):
        ct._rccl_world_size = 2
        src = torch.zeros((2, 4), dtype=torch.float16)

        with patch.object(ct, "_is_hipgraph_capture_active", return_value=False):
            with self.assertRaises(RuntimeError):
                ct._get_or_create_allgather_output(src)

        self.assertEqual(len(ct._hipgraph_allgather_outputs), 0)

    def test_set_hipgraph_capture_nccl_comm_clears_cache(self):
        ct._is_rocm_runtime = True
        ct._hipgraph_allgather_outputs[(tuple([2, 4]), torch.float16, "cpu", -1)] = (
            torch.zeros((2, 4), dtype=torch.float16)
        )
        fake_lib = object()

        with patch.object(ct, "_load_rccl", return_value=fake_lib), patch.object(
            ct, "_setup_rccl_api"
        ) as setup_api:
            ct.set_hipgraph_capture_nccl_comm(456, 4, 0)

        setup_api.assert_called_once_with(fake_lib)
        self.assertIsInstance(ct._rccl_comm, ctypes.c_void_p)
        self.assertEqual(ct._rccl_comm.value, 456)
        self.assertEqual(ct._rccl_world_size, 4)
        self.assertEqual(len(ct._hipgraph_allgather_outputs), 0)

    def test_enter_mode_without_handle_keeps_existing_comm(self):
        ct._is_rocm_runtime = True
        ct._rccl_comm = ctypes.c_void_p(123)
        ct._rccl_world_size = 4

        ct.enter_hipgraph_capture_mode(0, 0, 0)

        self.assertIsNotNone(ct._rccl_comm)
        self.assertEqual(ct._rccl_comm.value, 123)
        self.assertEqual(ct._rccl_world_size, 4)

    def test_set_hipgraph_capture_nccl_comm_zero_handle_clears_state(self):
        ct._is_rocm_runtime = True
        ct._rccl_comm = ctypes.c_void_p(123)
        ct._rccl_world_size = 4
        ct._hipgraph_allgather_outputs[(tuple([2, 4]), torch.float16, "cpu", -1)] = (
            torch.zeros((2, 4), dtype=torch.float16)
        )

        ct.set_hipgraph_capture_nccl_comm(0, 0, 0)

        self.assertIsNone(ct._rccl_comm)
        self.assertEqual(ct._rccl_world_size, 1)
        self.assertEqual(len(ct._hipgraph_allgather_outputs), 0)

    def test_prepare_hipgraph_capture_rccl_comm_bootstraps_on_tp(self):
        original_parallelism_config = ct._parallelism_config
        try:
            ct._is_rocm_runtime = True
            ct._parallelism_config = SimpleNamespace(tp_size=2, world_size=2)
            with patch.object(
                ct, "_bootstrap_hipgraph_capture_rccl_comm_from_tp_group"
            ) as bootstrap:
                ct._prepare_hipgraph_capture_rccl_comm_if_needed(ct._parallelism_config)
            bootstrap.assert_called_once()
        finally:
            ct._parallelism_config = original_parallelism_config

    def test_all_gather_fail_fast_when_capture_has_no_rccl_comm(self):
        ct._is_rocm_runtime = True
        ct._rccl_comm = None
        tensor = torch.zeros((1, 2), dtype=torch.float16)
        with patch.object(ct, "_is_hipgraph_capture_active", return_value=True):
            with self.assertRaises(RuntimeError):
                ct.all_gather(tensor, ct.Group.TP)


if __name__ == "__main__":
    unittest.main()
