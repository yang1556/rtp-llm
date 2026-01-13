import time

import pandas as pd
import torch
from async_grpc.network import AsyncGrpcClient, LLMRequest, RequestLog
from transformers import AutoTokenizer

datapath = "/home/admin/data/DAPO-Math-17k/data/dapo-math-17k.parquet"
model_path = "/home/admin/workspace/models/mtp/target_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

n = 4
batch_size = 32

client = AsyncGrpcClient(target_url="localhost:26001")
current_request_id = 0
df = pd.read_parquet(datapath)

enqueued_request_ids = []
for j in range(n):
    for i in range(batch_size):
        row = df.iloc[i]
        prompt = row["prompt"]
        raw_prompt = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,  # 添加生成提示（如assistant的起始token）
            tokenize=False,  # 只生成文本，不tokenize
        )
        encoded = tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False,  # 返回 Python list，而不是 tensor
        )
        input_ids = encoded["input_ids"][0]
        # print(input_ids.shape)
        seq_len = len(input_ids)

        attention_mask = encoded["attention_mask"]
        position_ids = torch.clip(
            torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None
        )  # 如果需要从 1 开始，用 range(1, seq_len + 1)

        client.enqueue(
            request=LLMRequest(
                tokens=input_ids.tolist(),
                rmp_tokens=input_ids.tolist(),
                request_id=current_request_id,
                parent_id=i,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                max_new_tokens=2048,
                n=1,
                timestamp=time.time(),
                attention_mask=attention_mask.tolist(),
                position_ids=position_ids.tolist(),
            )
        )
        enqueued_request_ids.append(current_request_id)
        current_request_id += 1


time_waited = 0
completed_requests: list[RequestLog] = []
stale_requests = []

start_time = time.time()

# batch_size = batch_size * sampling_param["n"]
# Wait until timeout or all enqueued requests are finished.
print(f"batch_size:{batch_size}, n: {n}")
while len(completed_requests) < n * batch_size:
    # print(f"completed_request:{len(completed_requests)}")
    for received_log in client.collect():
        if received_log.request.request_id in enqueued_request_ids:
            completed_requests.append(received_log)
            # if pending_requests:
            #     self.client.enqueue(pending_requests.pop(0))
        else:
            stale_requests.append(
                received_log
            )  # These are responses for requests not in the current batch.

end_time = time.time()
elapsed = end_time - start_time

print(f"total_time: {elapsed:.4f} s")
max_len = 0
for log in completed_requests:
    max_len = max(max_len, log.response.aux_info.output_len)

# print(log.response)
# print(completed_requests[2].response)
print(completed_requests[32].response.aux_info)
print(max_len)
# print(log.response.aux_info.output_len)
