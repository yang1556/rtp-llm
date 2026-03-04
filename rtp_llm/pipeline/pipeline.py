from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Union

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import DecodingState
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.ops import SpecialTokens, SpeculativeExecutionConfig, VitSeparation
from rtp_llm.pipeline.decode import (
    DecodeContext,
    decode_incremental_tokens,
    decode_non_incremental_tokens,
)
from rtp_llm.pipeline.decode import process_stop_id as decode_process_stop_id
from rtp_llm.pipeline.decode import process_stop_str as decode_process_stop_str
from rtp_llm.pipeline.sync_async import async_iterator_to_sync
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter
from rtp_llm.utils.word_util import get_stop_word_slices

request_counter = AtomicCounter()


class Pipeline(object):
    def __init__(
        self,
        special_tokens: SpecialTokens,  # SpecialTokens from ModelConfig
        pd_sep_config,  # PDSepConfig from ops
        addresses: list[str],  # RPC addresses for data parallel communication
        max_seq_len: int,  # max_seq_len_ from ModelConfig
        seq_size_per_block: int,  # seq_size_per_block_ from ModelConfig
        tokenizer: Optional[BaseTokenizer],
        sp_config: Optional[SpeculativeExecutionConfig] = None,
        mm_related_params: Optional[
            Any
        ] = None,  # mm_related_params from ModelConfig (optional)
        grpc_config: Optional[Any] = None,  # grpc_config from PyEnvConfigs (optional)
        vit_separation: Optional[VitSeparation] = None,  # Optional VitSeparation
    ):
        self.pd_sep_config = pd_sep_config
        self.tokenizer = tokenizer
        self._special_tokens: SpecialTokens = special_tokens
        self._mm_token: str = ""
        if mm_related_params:
            self._mm_token = mm_related_params.special_tokens.get(
                "default_mm_token", ""
            )

        self.backend_rpc_server_visitor = BackendRPCServerVisitor(
            max_seq_len=max_seq_len,
            seq_size_per_block=seq_size_per_block,
            pd_sep_config=pd_sep_config,
            addresses=addresses,
            sp_config=sp_config,
            grpc_config=grpc_config,
            vit_separation=vit_separation,
        )

    def encode(self, prompt: str):
        assert self.tokenizer is not None
        return self.tokenizer.encode(prompt)

    def decode(self, token_id: int):
        assert self.tokenizer is not None
        return self.tokenizer.decode([token_id])

    @staticmethod
    def create_generate_config(
        generate_config: Union[GenerateConfig, Dict[str, Any]],
        vocab_size: int,
        special_tokens: Any,
        tokenizer: BaseTokenizer,
        generate_env_config: Optional[Any] = None,
    ) -> GenerateConfig:
        if isinstance(generate_config, dict):
            config = GenerateConfig.create_generate_config(generate_config)
        else:
            # 认为是从frontend_worker传递进来的，不需要再处理一遍
            config = generate_config
        config.add_special_tokens(special_tokens)
        config.convert_select_tokens(vocab_size, tokenizer)
        config.add_thinking_params(tokenizer, generate_env_config)
        config.add_stop_ids_from_str(tokenizer)
        return config

    def __call__(
        self,
        prompt: str,
        urls: Optional[List[str]] = None,
        request_id: Optional[int] = None,
        generate_config: Optional[Union[GenerateConfig, Dict[str, Any]]] = None,
        generate_env_config: Optional[Any] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Iterator[GenerateResponse]:
        return self.pipeline(
            prompt,
            request_id=request_id,
            urls=urls,
            generate_config=generate_config,
            generate_env_config=generate_env_config,
            extra_config=extra_config,
        )

    def pipeline(
        self,
        prompt: str,
        request_id: Optional[int] = None,
        urls: Optional[List[str]] = None,
        generate_config: Optional[Union[GenerateConfig, Dict[str, Any]]] = None,
        generate_env_config: Optional[Any] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Iterator[GenerateResponse]:
        return async_iterator_to_sync(
            lambda: self.pipeline_async(
                prompt,
                request_id=request_id,
                urls=urls,
                generate_config=generate_config,
                generate_env_config=generate_env_config,
                extra_config=extra_config,
            )
        )

    @torch.inference_mode()
    def pipeline_async(  # type: ignore
        self,
        prompt: str,
        request_id: Optional[int] = None,
        urls: Optional[List[str]] = None,
        generate_config: Optional[Union[GenerateConfig, Dict[str, Any]]] = None,
        generate_env_config: Optional[Any] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[GenerateResponse, None]:
        begin_time = current_time_ms()

        if request_id is None:
            request_id = request_counter.increment()

        config_input = generate_config if generate_config is not None else {}
        if isinstance(config_input, dict) and extra_config:
            config_input = {**config_input, **extra_config}
        generate_config = self.create_generate_config(
            config_input,
            len(self.tokenizer),
            self._special_tokens,
            self.tokenizer,
            generate_env_config=generate_env_config,
        )
        mm_inputs = [MultimodalInput(url) for url in urls] if urls is not None else []

        if len(prompt) == 0:
            raise FtRuntimeException(
                ExceptionType.EMPTY_PROMPT_ERROR,
                "prompt should have at least one token!",
            )
        if type(prompt) is not str:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "expect string prompt, actual: " + str(prompt),
            )
        token_ids = self.tokenizer.encode(prompt)

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(
                generate_config.sp_advice_prompt
            )

        kmonitor.report(
            GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
        )
        kmonitor.report(GaugeMetrics.NUM_BEAMS_METRIC, generate_config.max_num_beams())
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(token_ids))
        return self.generate_stream(request_id, token_ids, mm_inputs, generate_config)

    @staticmethod
    def _merge_outputs_to_cache(
        cache: GenerateOutputs, new_batch: GenerateOutputs
    ) -> None:
        """Merge new_batch into cache: replace non-finished outputs with new step."""
        if not cache.generate_outputs:
            cache.generate_outputs = new_batch.generate_outputs
        else:
            cache.generate_outputs = [
                out if out.finished else new_batch.generate_outputs[i]
                for i, out in enumerate(cache.generate_outputs)
            ]
        assert len(cache.generate_outputs) == len(new_batch.generate_outputs)

    @staticmethod
    def _is_stream_finished(
        cache: GenerateOutputs, generate_config: GenerateConfig
    ) -> bool:
        """True when all outputs are finished and aux_info is requested."""
        return (
            all(o.finished for o in cache.generate_outputs) and generate_config.aux_info
        )

    @staticmethod
    def process_stop_id(
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
        tokens: Any,
        stop_word_ids: List[List[int]],
        stop_word_id_slices: List[List[int]],
    ) -> Any:
        """Delegate to decode module. Kept for tests and backward compatibility."""
        return decode_process_stop_id(
            generate_config,
            generate_output,
            tokens,
            stop_word_ids,
            stop_word_id_slices,
        )

    @staticmethod
    def process_stop_str(
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
        text: str,
        all_text: str,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        token_buffer: str,
    ):
        """Delegate to decode module. Kept for tests and backward compatibility."""
        return decode_process_stop_str(
            generate_config,
            generate_output,
            text,
            all_text,
            stop_word_str_list,
            stop_word_str_slices,
            token_buffer,
        )

    def decode_non_incremental_tokens(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[List[int]],
        stop_word_id_slices: List[List[int]],
        ouput_tokens_list: List[torch.Tensor],
    ):
        """Delegate to decode module. Kept for tests and backward compatibility."""
        ctx = DecodeContext(
            generate_config=generate_config,
            stop_word_str_list=stop_word_str_list,
            stop_word_str_slices=stop_word_str_slices,
            stop_word_ids=stop_word_ids,
            stop_word_id_slices=stop_word_id_slices,
            eos_token_id=self._special_tokens.eos_token_id,
        )
        return decode_non_incremental_tokens(
            ctx, generate_outputs, ouput_tokens_list, self.tokenizer
        )

    def decode_incremental_tokens(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[List[int]],
        stop_word_id_slices: List[List[int]],
        decoding_states: List[DecodingState],
        token_buffers: List[str],
        ouput_tokens_list: List[torch.Tensor],
    ):
        """Delegate to decode module. Kept for tests and backward compatibility."""
        ctx = DecodeContext(
            generate_config=generate_config,
            stop_word_str_list=stop_word_str_list,
            stop_word_str_slices=stop_word_str_slices,
            stop_word_ids=stop_word_ids,
            stop_word_id_slices=stop_word_id_slices,
            eos_token_id=self._special_tokens.eos_token_id,
        )
        return decode_incremental_tokens(
            ctx,
            generate_outputs,
            decoding_states,
            token_buffers,
            ouput_tokens_list,
            self.tokenizer,
        )

    @torch.inference_mode()
    async def generate_stream(
        self,
        request_id: int,
        token_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
    ) -> AsyncGenerator[GenerateResponse, None]:
        token_type_ids = []
        token_ids = torch.tensor(token_ids, dtype=torch.int)
        input = GenerateInput(
            request_id=request_id,
            token_ids=token_ids,
            mm_inputs=mm_inputs,
            generate_config=generate_config,
            tokenizer=self.tokenizer,
            token_type_ids=token_type_ids,
        )

        decode_ctx = DecodeContext(
            generate_config=generate_config,
            stop_word_str_list=generate_config.stop_words_str,
            stop_word_str_slices=get_stop_word_slices(generate_config.stop_words_str),
            stop_word_ids=generate_config.stop_words_list,
            stop_word_id_slices=get_stop_word_slices(generate_config.stop_words_list),
            eos_token_id=self._special_tokens.eos_token_id,
        )

        stream: AsyncGenerator[GenerateOutputs, None] = (
            await self.backend_rpc_server_visitor.enqueue(input)
        )

        decoding_states: List[DecodingState] = []
        ouput_tokens_list: List[torch.Tensor] = []
        token_buffers: List[str] = []
        generate_outputs_cache = GenerateOutputs()

        async for generate_outputs in stream:
            self._merge_outputs_to_cache(generate_outputs_cache, generate_outputs)
            begin_time = current_time_ms()
            is_incremental = (
                not generate_config.has_num_beams() and generate_config.is_streaming
            )
            if is_incremental:
                (
                    generate_texts,
                    output_lens,
                    decoding_states,
                    token_buffers,
                    ouput_tokens_list,
                ) = decode_incremental_tokens(
                    decode_ctx,
                    generate_outputs_cache,
                    decoding_states,
                    token_buffers,
                    ouput_tokens_list,
                    self.tokenizer,
                )
            else:
                (
                    generate_texts,
                    output_lens,
                    ouput_tokens_list,
                ) = decode_non_incremental_tokens(
                    decode_ctx,
                    generate_outputs_cache,
                    ouput_tokens_list,
                    self.tokenizer,
                )

            kmonitor.report(
                GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )
            yield GenerateResponse(
                generate_outputs=generate_outputs_cache, generate_texts=generate_texts
            )
            if self._is_stream_finished(generate_outputs_cache, generate_config):
                kmonitor.report(
                    GaugeMetrics.FT_ITERATE_COUNT_METRIC,
                    generate_outputs_cache.generate_outputs[0].aux_info.iter_count,
                )
                if len(output_lens) > 0:
                    kmonitor.report(
                        GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC,
                        sum(output_lens) / len(output_lens),
                    )
                break
