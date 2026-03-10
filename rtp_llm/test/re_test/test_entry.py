"""
lm_eval -> RTP-LLM ChatCompletionRequest 参数对应关系:

  simple_evaluate 参数            ->  RTP-LLM api_datatype.ChatCompletionRequest 字段
  -----------------------------     -----------------------------------------------
  model_args: model=...           ->  model
  model_args: base_url=...         ->  请求发往的 base URL（不进入 body）
  gen_kwargs: temperature=...     ->  temperature
  gen_kwargs: top_p=...            ->  top_p
  gen_kwargs: max_tokens=...      ->  max_tokens
  gen_kwargs: stop=...             ->  stop
  gen_kwargs: stream=...           ->  stream
  gen_kwargs: seed=...             ->  seed
  gen_kwargs: n=...                ->  n
  gen_kwargs 仅支持字符串，如 "temperature=0.7,max_tokens=1024"；不能传嵌套 dict，
  extra_configs 需通过请求头 X-RTP-Extra-Configs（JSON）传入，见 frontend_app.py。
"""

import os

os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import lm_eval

# lm_eval 要求 gen_kwargs 为字符串，不能为 dict
results = lm_eval.simple_evaluate(
    model="local-chat-completions",
    model_args="base_url=http://0.0.0.0:46000/v1,model=Qwen2-1.5B-Instruct",
    tasks=["gsm8k_cot"],
    apply_chat_template=False,
    gen_kwargs="temperature=0.7,top_p=1.0,max_tokens=1024",
)
print(results["results"])
