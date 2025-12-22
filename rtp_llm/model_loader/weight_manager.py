from __future__ import annotations

import logging
import re
from typing import Any, Mapping

import torch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights

# Assuming these imports are from your project and accessible
from rtp_llm.model_loader.weight_module import WeightModule

from .tipc import TensorIPCMeta, TensorTransportServer


class WeightManager:
    """
    Manages model weight updates, including renaming weights from an external
    source and handling inter-process communication (IPC) for tensor transfer.
    It ensures that incoming tensors are correctly processed and sharded/replicated
    as per the rtp-llm model's internal structure (e.g., for Tensor Parallelism (TP)
    or Pipeline Parallelism (PP)).
    """

    def __init__(self, engine: BaseEngine) -> None:
        """
        Initializes the WeightManager with an BaseEngine instance.
        Args:
            model: An instance of `BaseEngine` containing the model's structure,
                   device information, and weight loaders.
        Error Handling:
            This constructor does not explicitly raise errors, but relies on the
            correct initialization of the `BaseEngine` and its internal components.
            Issues with `model` object structure could lead to `AttributeError`.
        """
        self._engine: BaseEngine = engine
        self._device: torch.device = engine.model.device
        self._weights: ModelWeights = engine.model.weight
        self._weights_loader: ModelLoader = engine.model.model_weights_loader
        self._weight_module = self._weights_loader._model_weights_info

        if engine.propose_model is not None:
            self._propose_weights: ModelWeights = engine.propose_model.model.weight
            self._propose_weights_loader: ModelLoader = (
                engine.propose_model.model.model_weights_loader
            )
            self._propose_weights_module = (
                self._propose_weights_loader._model_weights_info
            )
            for layer in self._propose_weights_module.layer_weights:
                for receptor in layer:
                    print(f"layer weights:{receptor.name}")
            for weight in self._propose_weights_module.weights:
                print(f"model weigths: {weight.name}")

    def extract_layer_number(self, s: str) -> int | None:
        """
        Extracts the layer number (an integer) from a string that follows
        the pattern 'layers.<number>'.
        Args:
            s: The input string, e.g., 'model.layers.2.mlp.gate_proj.weight'.
        Returns:
            The extracted layer number as an integer if found; otherwise, returns `None`.
        Error Handling:
            Returns `None` if the pattern 'layers.<number>' is not found,
            or if the captured group cannot be converted to an integer.
        """
        match = re.search(r"layers\.(\d+)", s)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        else:
            return None

    def mount(self, name: str, tensor: torch.Tensor, is_propose: bool) -> None:

        if is_propose:
            if (
                self._propose_weights is None
                or self._propose_weights_loader is None
                or self._propose_weights_module is None
            ):
                raise RuntimeError(
                    "Propose model components are not initialized for weight update."
                )
            target_weights = self._propose_weights
            target_loader = self._propose_weights_loader
            target_module = self._propose_weights_module
        else:
            target_weights = self._weights
            target_loader = self._weights_loader
            target_module = self._weight_module
        tensor = tensor.to(self._device)
        config = target_loader.get_load_config()

        if "layers" in name:
            # This is a layer-specific weight
            layer_id: int | None = self.extract_layer_number(name)
            if layer_id is None:
                raise ValueError(
                    f"Invalid layer weight name format: '{name}'. "
                    "Could not extract layer number. Expected format like 'model.layers.<id>...'"
                )
            if layer_id > len(target_module.layer_weights):
                raise IndexError("layer index out of range.")

            fail: bool = True
            for receptor in target_module.layer_weights[layer_id]:
                if name.startswith(f"model.layers.{layer_id}.{receptor.name}"):
                    # 这里需要使用 start with 判断, 因为可以出现 ffn_weights, moe_weights 这样的组合权重
                    # 这些权重在 rtp 里面的名字是 model.layers.0.__ffn_weights__
                    # 但输入的权重是 model.layers.0.__ffn_weights__.intermediate_weights
                    # 这些权重需要被对应的 receptor 处理

                    assert isinstance(receptor, WeightModule)
                    # split tensor into shards
                    _config = config.copy()
                    _config.use_stack_weight = True

                    shard = receptor.update(
                        tensor=tensor,
                        device=self._device,
                        load_config=_config,
                        module_name=name,
                    )
                    if isinstance(shard, dict):
                        shard = next(iter(shard.values()))

                    if "__ffn_weights__" == receptor.name:
                        # 这里需要按照一定规则更换输入的权重名字
                        name = name.replace("__ffn_weights__", "ffn_weights")
                        name = name[name.find("ffn_weights") :]
                        target_weights.update_layer_weight(
                            layer_id=layer_id,
                            name=name,
                            data=shard,
                            is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                        )
                    elif "__moe_weights__" == receptor.name:
                        # 这里需要按照一定规则更换输入的权重名字
                        name = name.replace("__moe_weights__", "partial_moe_weights")
                        name = name[name.find("partial_moe_weights") :]
                        target_weights.update_layer_weight(
                            layer_id=layer_id,
                            name=name,
                            data=shard,
                            is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                        )
                    else:
                        # update tensor weight
                        target_weights.update_layer_weight(
                            layer_id=layer_id,
                            name=receptor.name,
                            data=shard,
                            is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                        )
                    fail = False

            if fail:
                raise KeyError(
                    f"{name} not found. wanted name list is {[f'model.layers.{layer_id}.{w.name}' for w in target_module.layer_weights[layer_id]]}"
                )

        else:
            # weight is global weight
            fail: bool = True
            for weight in target_module.weights:
                if f"model.{weight.name}" == name:
                    shard: dict = weight.update(
                        tensor,
                        self._device,
                        load_config=target_loader.get_load_config(),
                    )
                    if isinstance(shard, dict):
                        shard = next(iter(shard.values()))
                    target_weights.update_global_weight(
                        name=weight.name,
                        data=shard,
                        is_master=(config.dp_rank == 0 and config.tp_rank == 0),
                    )
                    fail = False

            if fail:
                raise KeyError(
                    f"{name} not found. wanted name list is {[f'model.{w.name}' for w in target_module.weights]}"
                )

        torch.cuda.synchronize()
        logging.info(f"RtpLLM Finish Weights Update: {name}.")

    def update(self, req: Mapping[str, Any]) -> None:
        """
        Receives an Inter-Process Communication (IPC) tensor description and
        updates the corresponding model weights.
        For models with Tensor Parallelism (TP) or Pipeline Parallelism (PP),
        this function expects the transmitted tensor to be a complete, unsharded tensor.
        It then handles the internal sharding or replication according to the
        rtp-llm's specific model parallelism configuration.
        Args:
            req: A dictionary containing the IPC request details. Expected keys are:
                 - "desc": A list of string describing the tensor's IPC metadatas
                           (e.g., `CuIpcTensorMeta` or `SharedMemIpcMeta` encoded string).
                 - "method": A string indicating the IPC method used ("cuda_ipc" or "shm").
        Returns:
            None. The method updates internal model weights directly.
        Error Handling:
            - `KeyError`: If "desc", or "method" fields are missing from `req`.
            - `ValueError`: If the "method" is invalid (not "cuda_ipc" or "shm"),
                            or if a layer weight name is invalid and its ID cannot be extracted.
            - `Exception`: If the tensor cannot be built from the IPC metadata (e.g., invalid descriptor).
                          This is a general catch-all for unexpected failures in `_t_helper.build_from_meta`.
        """
        # --- Validate Request Fields ---
        if "desc" not in req:
            raise KeyError(
                "Update request is missing the 'desc' field. "
                "It must contain IPC tensor metadata."
            )
        if "method" not in req:
            raise KeyError(
                "Update request is missing the 'method' field. "
                "It must specify the IPC method (e.g., 'cuda_ipc' or 'shm')."
            )
        if "storage" not in req:
            raise KeyError("Update request is missing the 'storage' field. ")
        if "device" not in req:
            raise KeyError("Update request is missing the 'device' field. ")
        method: str = str(req["method"])
        storage: str = str(req["storage"])
        desc: list[str] = req["desc"]
        device: str = req["device"]
        target_type: str = req["target_model_type"]

        if target_type == "propose":
            is_propose = True
        elif target_type == "main":
            is_propose = False
        else:
            raise KeyError("target_model_type must be main or propose. ")

        reader = TensorTransportServer(method=method, storage=storage)
        metas = [TensorIPCMeta.decode(content) for content in desc]
        tensors = reader.read(metas, device)

        for m, t in zip(metas, tensors):
            logging.info(f"Ipc received tensor: {m.name}, {t.shape}, {t.dtype}")
            self.mount(m.name, t, is_propose)
