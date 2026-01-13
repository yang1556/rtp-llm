# simplified_test_rtp_client.py
import asyncio
import glob
import os
import unittest
from time import time
from typing import Any, Dict

import httpx
import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from tipc import TensorTransportClient

# PATH = "/root/hf/Qwen3-30B-A3B"
datapath = "/home/admin/data/DAPO-Math-17k/data/dapo-math-17k.parquet"
# PATH = "/home/admin/workspace/models/mtp/target_model"
PATH = "/home/admin/workspace/models/mtp/draft_model"
PROPOSEPATH = "/home/admin/workspace/hzy/verl/checkpoints/verl_example/qwen14_mtp_train/global_step_5/draft"

# PATH = "/mnt/nas1/hf/Qwen3-8B"
Device: int = 0
METHOD = "cuipc"


class RtpLLMHttpClient(TensorTransportClient):
    def __init__(self, address: str, frentend_port: int, backend_port: int):
        super().__init__(
            device_id=Device,
            url=f"http://{address}:{backend_port}/update_weight",
            method=METHOD,
        )
        self.client1 = httpx.AsyncClient(
            base_url=f"http://{address}:{frentend_port}", timeout=30.0
        )
        self.client2 = httpx.AsyncClient(
            base_url=f"http://{address}:{backend_port}", timeout=30.0
        )
        self.records: Dict[str, Any] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client1.aclose()
        await self.client2.aclose()

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        response.raise_for_status()
        response = response.json()

        if response is not None and "status" in response:
            if response["status"] == "error":
                raise Exception(f"server internal error: {response}")
            if "error" in response:
                raise Exception(f"server request error: {response}")
        return response

    async def chat_completion(self, name: str, prompt: str) -> None:
        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.9,
            "topk": 50,
            "aux_info": True,
            "stream": False,
        }
        response = await self.client1.post("/v1/chat/completions", json=payload)
        # print(response.json())
        # print(response.json()["choices"][0]["message"]['content'])
        print(response.json()["aux_info"], response.json()["aux_info"]["output_len"])
        content = self._handle_response(response)
        self.records[name] = {
            "content": content["choices"][0]["message"],
            "end_time": time(),
        }

    async def detach(self) -> None:
        response = await self.client2.post("/detach_physical_memory")
        self._handle_response(response)
        print(f"detach server memory: {response}")

    async def attach(self) -> None:
        response = await self.client2.post("/attach_physical_memory")
        self._handle_response(response)
        print(f"attach server memory: {response}")

    async def update_model_weight(self, path: str, target_model: str):
        files = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        print(files)
        weights = {}
        for fn in tqdm(files, desc="loading weights"):
            part = load_file(fn, device="cpu")
            weights.update(part)

        # sort weight is necessary
        weights = [(name, tensor) for name, tensor in tqdm(weights.items())]
        for name, tensor in tqdm(sorted(weights), "updating weights"):
            print(name)
            if target_model == "propose" and name == "model.norm.weight":
                continue
            # if target_model =="propose" and name == "model.layers.0.eh_proj.weight":
            #     continue
            # if target_model =="propose" and name == "lm_head.weight":
            #     continue
            # if target_model =="propose" and name == "model.embed_tokens.weight":
            #     continue
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)

            self.write(name, tensor.to(f"cuda:{Device}"), target_model)
        self.flush(named_tensors=None, target_model_type=target_model)

    async def pause(self) -> None:
        response = await self.client2.post("/pause")
        self._handle_response(response)

    async def restart(self) -> None:
        response = await self.client2.post("/restart")
        self._handle_response(response)


class TestRtpClient(unittest.IsolatedAsyncioTestCase):
    async def test_full_flow(self):
        async with RtpLLMHttpClient("localhost", 26000, 26006) as client:
            await client.attach()
            prompt = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.In triangle $ABC$, $\sin \angle A = \frac{4}{5}$ and $\angle A < 90^\circ$. Let $D$ be a point outside triangle $ABC$ such that $\angle BAD = \angle DAC$ and $\angle BDC = 90^\circ$. Suppose that $AD = 1$ and that $\frac{BD}{CD} = \frac{3}{2}$. If $AB + AC$ can be expressed in the form $\frac{a\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.Remember to put your answer on its own line after "Answer:". '
            df = pd.read_parquet(datapath)
            for i in range(1):
                row = df.iloc[i]
                # print(row)
                for j in range(4):
                    prompt = row["prompt"]  # 一个 list
                    first_item = prompt[0]  # 第一个 dict
                    content = first_item["content"]
                    await client.chat_completion("chat_1", content)
            # await client.pause()
            # await client.attach()
            # await client.update_model_weight(path=PATH, target_model="propose")
            # #await client.detach()
            # #await client.restart()

            # #await client.attach()
            # await client.chat_completion("chat_1", prompt)


if __name__ == "__main__":
    unittest.main()
    # df = pd.read_parquet(datapath)
    # row = df.iloc[0]
    # print(row)
    # prompt = row["prompt"]          # 一个 list
    # first_item = prompt[0]          # 第一个 dict
    # content = first_item["content"] # 取 content
    # "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.In triangle $ABC$, $\sin \angle A = \frac{4}{5}$ and $\angle A < 90^\circ$. Let $D$ be a point outside triangle $ABC$ such that $\angle BAD = \angle DAC$ and $\angle BDC = 90^\circ$. Suppose that $AD = 1$ and that $\frac{BD}{CD} = \frac{3}{2}$. If $AB + AC$ can be expressed in the form $\frac{a\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.Remember to put your answer on its own line after \"Answer:\". "
    # print(content)
