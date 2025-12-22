import logging
import re
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import requests  # Import the requests library for making HTTP requests
import torch

from .core import COMMON_PREFIX, MethodType, TensorIPCMeta
from .ffi import NvIpcWriter, NvShmWriter, TipcLib


@dataclass
class NamedTensor:
    """
    A simple data container for associating a name (string) with a PyTorch tensor.

    Attributes:
        name (str): The name of the tensor (e.g., a weight name in a model).
        tensor (torch.Tensor): The PyTorch tensor data.
    """

    name: str
    tensor: torch.Tensor

    def __str__(self):
        return f"{self.name}[{self.tensor.shape}]"

    def __repr__(self):
        return self.__str__()


def get_expert_id(name: str) -> int:
    try:
        extracted = name[name.find("experts.") + len("experts.") :]
        extracted = extracted[: extracted.find(".")]
        return int(extracted)
    except Exception as e:
        raise Exception(f"fail to get experts id from given weight name: {name}")


def preprocess(nt: NamedTensor) -> NamedTensor:
    # convert nt.name(huggingface) to rtp-llm name
    REPLACEMENT = {
        "model.embed_tokens.weight": "model.embedding",
        "input_layernorm.weight": "pre_layernorm_weights.gamma",
        "input_layernorm.bias": "pre_layernorm_weights.beta",
        "post_attention_layernorm.weight": "post_layernorm_weights.gamma",
        "self_attn.o_proj.weight": "self_attention_weights.attention_output_weight.kernel",
        "self_attn.o_proj.bias": "self_attention_weights.attention_output_weight.bias",  # error.
        "self_attn.qkv_proj.weight": "self_attention_weights.query_weight.kernel",
        "self_attn.qkv_proj.bias": "self_attention_weights.query_weight.bias",
        "model.norm.weight": "model.final_layernorm.gamma",
        "self_attn.k_norm.weight": "self_attention_weights.k_layernorm.gamma",
        "self_attn.q_norm.weight": "self_attention_weights.q_layernorm.gamma",
        "self_attn.k_norm.bias": "self_attention_weights.k_layernorm.beta",
        "self_attn.q_norm.bias": "self_attention_weights.q_layernorm.beta",
        # mtp
        "e_norm.weight": "multi_tokens_predict_enorm.weight",
        "eh_proj.weight": "multi_tokens_predict_eh_proj.weight",
        "h_norm.weight": "multi_tokens_predict_hnorm.weight",
        "final_head.norm.weight": "multi_tokens_predict_final_layernorm.gamma",
    }
    for k, v in REPLACEMENT.items():
        if k in nt.name:
            nt.name = nt.name.replace(k, v)
            break

    if nt.name == "lm_head.weight":
        nt.name = "model.lm_head"

    # tensor preprocess function, convert hugging face layout to rtp-llm layout
    if "self_attention_weights.attention_output_weight.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "self_attention_weights.query_weight.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "__ffn_weights__.intermediate_weight13.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "__ffn_weights__.intermediate_weight2.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)

    """
    if "__moe_weights__.intermediate_weight.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "__moe_weights__.intermediate_weight2.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(1, 2)
    """
    if "__moe_weights__.gate.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    return nt


class TensorBucketBuilder:
    """
    Groups and combines model weights, particularly those that are stored
    in a fragmented manner (like q_proj, k_proj, v_proj, or MoE expert weights)
    in the training system, before transferring them to the inference system.
    This helps in reducing the number of IPC calls and potentially makes the
    weights compatible with the inference system's expected format.
    """

    def __init__(self):
        """
        Initializes the TensorBucketBuilder with empty buffers for pending tensors
        and those ready to be sent, and sets the current layer ID to "Undefined".
        """
        self.pending_buffer: list[NamedTensor] = []
        self.bucket: list[NamedTensor] = []
        self.layer_id: str = "Undefined"

    def get_layer_id(self, name: str) -> str | None:
        """
        Extracts the layer ID from a tensor's name using a regular expression.

        Args:
            name (str): The name of the tensor.

        Returns:
            str | None: The extracted layer ID (e.g., 'model.layers.0.') or None if no match is found.
        """
        pattern = r"(model\.layers\.\d+\.)"
        match = re.search(pattern, name)

        if match:
            layer_id = match.group(1)
            return layer_id
        else:
            return None

    def flush(self) -> list[NamedTensor]:
        """
        Processes and combines the tensors in the pending buffer based on
        predefined rules (e.g., combining q_proj, k_proj, v_proj into qkv_proj,
        or grouping MoE expert weights), clears both buffers, and returns
        the list of combined and uncombined (but ready-to-send) NamedTensors.

        Returns:
            list[NamedTensor]: A list of NamedTensor objects ready for transport.
        """
        # Mainly combine three cases
        # One is q, k, v proj need to be combined into qkv_proj
        # One is feedforward a, b, c needs to be combined into feedforward a, c

        # One is expert.xx.down_proj needs to be combined into partial_moe_weights.intermediate_weight.kernel
        # One is expert.xx.up_proj needs to be combined into partial_moe_weights.intermediate_weight2.kernel
        # One is expert.xx.gate_proj needs to be combined into partial_moe_weights.gate.kernel

        qkv_weight: list[NamedTensor] = []
        qkv_bias: list[NamedTensor] = []
        expert_down_proj: list[NamedTensor] = []
        expert_up_proj: list[NamedTensor] = []
        expert_gate_proj: list[NamedTensor] = []
        mlp_weight: list[NamedTensor] = []

        for nt in self.pending_buffer:
            # qkv proj
            if ".q_proj.weight" in nt.name:
                qkv_weight.append(nt)
            if ".k_proj.weight" in nt.name:
                qkv_weight.append(nt)
            if ".v_proj.weight" in nt.name:
                qkv_weight.append(nt)

            if ".q_proj.bias" in nt.name:
                qkv_bias.append(nt)
            if ".k_proj.bias" in nt.name:
                qkv_bias.append(nt)
            if ".v_proj.bias" in nt.name:
                qkv_bias.append(nt)

            # expert
            if ".experts." in nt.name:
                if "down_proj" in nt.name:
                    expert_down_proj.append(nt)
                if "up_proj" in nt.name:
                    expert_up_proj.append(nt)
                if "gate_proj" in nt.name:
                    expert_gate_proj.append(nt)
            elif ".mlp." in nt.name:
                mlp_weight.append(nt)

        ret: list[NamedTensor] = self.bucket.copy()
        if len(qkv_weight) != 0:
            if len(qkv_weight) != 3:
                raise Exception(
                    f"qkv proj 的权重个数错误，应该有 3个 名为 q_proj, k_proj, v_proj 的权重，但接收到 {qkv_weight}"
                )
            q = [nt for nt in qkv_weight if "q_proj.weight" in nt.name]
            k = [nt for nt in qkv_weight if "k_proj.weight" in nt.name]
            v = [nt for nt in qkv_weight if "v_proj.weight" in nt.name]
            ret.append(
                NamedTensor(
                    name=self.layer_id + "self_attn.qkv_proj.weight",
                    tensor=torch.cat([nt.tensor for nt in q + k + v], dim=0),
                )
            )

        if len(qkv_bias) != 0:
            if len(qkv_bias) != 3:
                raise Exception(
                    f"qkv proj 的权重个数错误，应该有 3个 名为 q_proj, k_proj, v_proj 的权重，但接收到 {qkv_bias}"
                )
            q = [nt for nt in qkv_bias if "q_proj.bias" in nt.name]
            k = [nt for nt in qkv_bias if "k_proj.bias" in nt.name]
            v = [nt for nt in qkv_bias if "v_proj.bias" in nt.name]

            ret.append(
                NamedTensor(
                    name=self.layer_id + "self_attn.qkv_proj.bias",
                    tensor=torch.cat([nt.tensor for nt in q + k + v], dim=0),
                )
            )

        if len(expert_up_proj) != 0 and len(expert_gate_proj) != 0:
            up_projs = sorted(expert_up_proj, key=lambda x: get_expert_id(x.name))
            gt_projs = sorted(expert_gate_proj, key=lambda x: get_expert_id(x.name))

            tensors = []
            for up, gt in zip(up_projs, gt_projs):
                tensors.append(torch.cat([up.tensor, gt.tensor], dim=0).unsqueeze(0))

            ret.append(
                NamedTensor(
                    name=self.layer_id + "__moe_weights__.intermediate_weight.kernel",
                    tensor=torch.cat(tensors, dim=0),
                )
            )

        if len(expert_down_proj) != 0:
            ret.append(
                NamedTensor(
                    name=self.layer_id + "__moe_weights__.intermediate_weight2.kernel",
                    tensor=torch.cat(
                        [
                            nt.tensor.unsqueeze(0)
                            for nt in sorted(
                                expert_down_proj, key=lambda x: get_expert_id(x.name)
                            )
                        ],
                        dim=0,
                    ),
                )
            )

        if len(mlp_weight) > 0:
            if len(mlp_weight) == 1:
                # moe gate
                nt = mlp_weight[0]
                ret.append(
                    NamedTensor(
                        name=self.layer_id + "__moe_weights__.gate.kernel",
                        tensor=nt.tensor,
                    )
                )

            elif len(mlp_weight) == 3:
                up_proj = [nt for nt in mlp_weight if "up_proj.weight" in nt.name][
                    0
                ].tensor
                dn_proj = [nt for nt in mlp_weight if "down_proj.weight" in nt.name][
                    0
                ].tensor
                gt_proj = [nt for nt in mlp_weight if "gate_proj.weight" in nt.name][
                    0
                ].tensor

                ret.append(
                    NamedTensor(
                        name=self.layer_id
                        + "__ffn_weights__.intermediate_weight13.kernel",
                        tensor=torch.cat([gt_proj, up_proj], dim=0),
                    )
                )
                ret.append(
                    NamedTensor(
                        name=self.layer_id
                        + "__ffn_weights__.intermediate_weight2.kernel",
                        tensor=dn_proj,
                    )
                )
            else:
                raise Exception(f"mlp 的权重个数错误，接收到 {mlp_weight}")

        self.pending_buffer.clear()
        self.bucket.clear()
        return ret

    def combine_layer_tensor(
        self, name: str, tensor: torch.Tensor
    ) -> list[NamedTensor]:
        """
        Receives a tensor and either adds it to a pending buffer for later combination,
        adds it to a bucket for immediate sending, or flushes the pending tensors
        if a new layer's tensor is encountered.

        Args:
            name (str): The name of the tensor.
            tensor (torch.Tensor): The tensor data.

        Returns:
            list[NamedTensor]: A list of combined/ready-to-send tensors. This list is empty
                               unless a flush operation was triggered (by a layer ID change)
                               or if the tensor had no layer ID.
        """
        # Call this combine function, which will try to concatenate weights by layer id and send them,
        # which is usually faster and can solve some weight concatenation problems.
        INTRESTED_PREFIXS = [".q_proj.", ".k_proj.", ".v_proj.", ".experts.", ".mlp."]

        layer_id: str | None = self.get_layer_id(name)
        if layer_id is None:
            # If this is a weight without a layer id, we choose to send it individually
            return [NamedTensor(name, tensor)]

        ret: list[NamedTensor] = []
        if layer_id != self.layer_id:
            # If the layer ID changes, flush the current pending buffer and bucket
            ret = self.flush()
            self.layer_id = layer_id

        if any([prefix in name for prefix in INTRESTED_PREFIXS]):
            # If this is a tensor that needs to be concatenated, put it into the pending buffer
            self.pending_buffer.append(NamedTensor(name, tensor))

        else:
            # If this is a weight that does not need to be concatenated, put it directly into the bucket
            self.bucket.append(NamedTensor(name, tensor))

        return ret


class TensorTransportClient:
    """
    Client for efficiently transporting tensor data to a remote server.

    Features:
    - Supports two transport methods: shared memory ("shm") and CUDA IPC ("cuipc")
    - For "shm", creates a single large persistent shared memory buffer that can be reused
    """

    def __init__(
        self,
        device_id: int,
        method: MethodType = "shm",
        buffer_size: int = 4 * 1024 * 1024 * 1024,
        url: str = "http://localhost:26006/update_weight",
    ):
        """
        Initialize TensorTransportClient and create transport resources
        according to the selected method.

        Args:
            device_id (int):
                Target GPU device ID when using CUDA IPC ("cuipc").
                For "shm" mode this is not used but kept for a unified interface.
            method (str, optional):
                Transport method:
                  - "shm": use system shared memory + NvShmWriter
                  - "cuipc": use CUDA IPC + NvIpcWriter
                Defaults to "shm".
            buffer_size (int, optional):
                Maximum buffer size in bytes to pre-allocate.
                For "shm": size of the shared memory segment.
                For "cuipc": internal buffer size of NvIpcWriter.
                Default is 4GB.
            url (str, optional):
                HTTP endpoint used to notify the server about shared memory
                or IPC handle information.
                Default is "http://localhost:26006/update_weight".

        Raises:
            ValueError: If method is invalid or buffer_size is non-positive.
            RuntimeError: If creation of shared memory or IPC writer fails.
        """

        # Basic argument validation
        if method not in ("shm", "cuipc"):
            raise ValueError(
                f"Unsupported method '{method}', expected 'shm' or 'cuipc'."
            )

        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}.")

        # Normalize URL: automatically add scheme if user passed "localhost:26006/..."
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url

        self.device_id = device_id
        self.cuda_device_pcie: str | None = None
        self.method: MethodType = method
        self.buffer_size = buffer_size
        self.url = url

        # Container/builder for tensors to be transported
        self.tensor_bucket = TensorBucketBuilder()

        # Shared memory and writer objects; will be initialized depending on method
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.storage: Optional[str] = None
        self.writer: Optional[NvIpcWriter | NvShmWriter] = None

        # Initialize underlying transport resource based on selected method
        if method == "shm":
            self._init_shared_memory_writer()
        else:  # method == "cuipc"
            self._init_ipc_writer()

    def _init_shared_memory_writer(self) -> None:
        """
        Initialize a shared memory segment and its corresponding NvShmWriter.
        """
        # Generate a unique shared memory name to avoid collisions
        raw_name = f"{COMMON_PREFIX}_persistent_{uuid.uuid4()}"
        try:
            shm = shared_memory.SharedMemory(
                create=True,
                size=self.buffer_size,
                name=raw_name,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create shared memory '{raw_name}': {e}"
            ) from e

        self.shm = shm
        self.storage = f"/dev/shm/{raw_name}"

        try:
            self.writer = TipcLib.NvShmWriter(self.storage)
        except Exception as e:
            # If writer creation fails, clean up the shared memory segment
            try:
                self.shm.close()
                self.shm.unlink()
            finally:
                self.shm = None
                self.shm_name = None
            raise RuntimeError(
                f"Failed to create NvShmWriter for '{self.shm_name}': {e}"
            ) from e

    def _init_ipc_writer(self) -> None:
        """
        Initialize a CUDA IPC writer (NvIpcWriter).
        """
        try:
            self.writer = TipcLib.NvIpcWriter(
                device=self.device_id,
                buffer_size=self.buffer_size,
            )
            self.storage = self.writer.build().hex()

        except Exception as e:
            raise RuntimeError(
                f"Failed to create NvIpcWriter on device {self.device_id}: {e}"
            ) from e

    def close(self) -> None:
        """Explicitly release underlying resources (writer and shared memory)."""
        if self.writer is not None:
            try:
                self.writer.close()
            finally:
                self.writer = None

        # Close and unlink shared memory if it was created
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            finally:
                self.shm = None
                self.shm_name = None

    def __del__(self):
        """
        Destructor: attempt to free resources when the object is garbage-collected.
        Exceptions are suppressed to avoid errors during interpreter shutdown.
        """
        try:
            self.close()
        except Exception:
            pass

    def build_tensor_meta(self, name: str, tensor: torch.Tensor) -> TensorIPCMeta:
        m: TensorIPCMeta = TensorIPCMeta(
            name=name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            size=tensor.element_size() * tensor.numel(),
        )
        return m

    def _send(self, encoded_metas: list[TensorIPCMeta], target_model_type: str) -> None:
        """
        Internal method to send the IPC metadata to the server via an HTTP POST request.

        Args:
            encoded_metas (list[SharedMemIpcMeta]): A list of metadata objects
                                                    describing the tensors' locations in SHM.

        Raises:
            requests.exceptions.RequestException: If there's an error during the HTTP request.
            IOError: If the server returns a non-200 status code.
            Exception: If the server returns a 'status': 'error' in its JSON response.
        """

        from time import time

        if not encoded_metas:
            return

        try:
            response = requests.post(
                self.url,
                json={
                    "time": time(),
                    "desc": [m.encode() for m in encoded_metas],
                    "method": self.method,
                    "storage": self.storage,
                    "device": self.cuda_device_pcie,
                    "target_model_type": target_model_type,
                },
            )

            if response.status_code == 200:
                response_ = response.json()

                if "status" in response_:
                    if response_["status"] == "error":
                        raise Exception(
                            f"IPC Transport failed, Server returns error: {response_}"
                        )
            else:
                raise IOError(
                    f"IPC Tranport failed, Server returns code: {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            raise e

        except Exception as e:
            raise e

    def write(self, name: str, t: torch.Tensor, target_model_type: str):
        """
        Processes a tensor, potentially combining it with others, writes the combined
        tensors' data into the pre-allocated shared memory block, and then sends
        the metadata to the remote host via an HTTP POST request.

        The tensor is sent layer by layer.

        The tensor is first converted to a contiguous CPU tensor before processing.

        Args:
            name (str): The name to associate with the tensor.
            t (torch.Tensor): The tensor to send.

        Raises:
            requests.exceptions.RequestException: If there's an error during the HTTP request.
            ValueError: If an unsupported IPC method is chosen or if the tensor is too large for SHM.
        """

        logging.info(
            f"tipc transporting tensor, name={name}, dtype={t.dtype}, shape={t.shape}, device={t.device}"
        )

        if self.cuda_device_pcie is None:
            self.cuda_device_pcie = TipcLib.get_tensor_device_pcie_str(t)
        if t.is_cuda and self.cuda_device_pcie != TipcLib.get_tensor_device_pcie_str(t):
            print(self.cuda_device_pcie, TipcLib.get_tensor_device_pcie_str(t))
            raise RuntimeError("Tensor has different cuda device.")

        # Combine the current tensor with any pending tensors from the same layer
        named_tensors = self.tensor_bucket.combine_layer_tensor(name, t)

        if len(named_tensors) > 0:
            self.flush(named_tensors, target_model_type)

    def flush(
        self,
        named_tensors: list[NamedTensor] | None = None,
        target_model_type: str = None,
    ):
        """
        Forces the TensorBucketBuilder to process and send any remaining
        tensors in its buffers, regardless of layer ID changes.
        """
        if named_tensors is None:
            named_tensors = self.tensor_bucket.flush()

        if self.writer is None:
            raise RuntimeError("Internal Error.")

        if len(named_tensors) > 0:
            encoded_metas: list[TensorIPCMeta] = []

            for nt in named_tensors:
                nt = preprocess(nt)

                nt.tensor = nt.tensor.contiguous()
                m: TensorIPCMeta = self.build_tensor_meta(
                    name=nt.name, tensor=nt.tensor
                )
                self.writer.write(nt.tensor)
                encoded_metas.append(m)

            # necessary synchronize here.
            torch.cuda.synchronize()
            self._send(encoded_metas=encoded_metas, target_model_type=target_model_type)
            self.writer.reset()
