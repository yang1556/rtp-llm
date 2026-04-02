import json
from typing import List

from rtp_llm.openai.api_datatype import ChatCompletionRequest, ContentPartTypeEnum
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.ops import MMPreprocessConfig
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Qwen35Renderer(Qwen3CoderRenderer):
    def _render_messages(
        self, request: ChatCompletionRequest, add_vision_id: bool
    ) -> PromptWithMMInput:
        urls = []
        types = []
        preprocess_configs = []
        final_messages = []
        for message in request.messages:
            msg_dict = {"role": message.role.value}

            if isinstance(message.content, list):
                now_content = []
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        now_content.append({"type": "text", "text": content_part.text})
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        urls.append(content_part.image_url.url)
                        types.append(MMUrlType.IMAGE)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                        now_content.append(
                            {"type": "image", "image": content_part.image_url.url}
                        )
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url != None
                        urls.append(content_part.video_url.url)
                        types.append(MMUrlType.VIDEO)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                        now_content.append(
                            {"type": "video", "video": content_part.video_url.url}
                        )
                msg_dict["content"] = now_content
            else:
                msg_dict["content"] = message.content

            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            final_messages.append(msg_dict)

        final_tools = []
        if request.tools:
            for tool in request.tools:
                final_tools.append(
                    {
                        "type": tool.type,
                        "function": tool.function.model_dump(
                            exclude_none=True, mode="json"
                        ),
                    }
                )

        prompt = self.tokenizer.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=add_vision_id,
            tools=final_tools,
        )

        return PromptWithMMInput(
            prompt=prompt,
            urls=urls,
            mm_types=types,
            preprocess_configs=preprocess_configs,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        prompt_and_mm_input = self._render_messages(
            request,
            request.extra_configs.add_vision_id if request.extra_configs else True,
        )
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("qwen3_vl_moe", Qwen35Renderer)
register_renderer("qwen3_vl", Qwen35Renderer)
register_renderer("qwen35_moe", Qwen35Renderer)
register_renderer("qwen35_dense", Qwen35Renderer)
