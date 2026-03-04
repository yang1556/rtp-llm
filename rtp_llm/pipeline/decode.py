"""Stream decode and stop-word handling for pipeline generation."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs
from rtp_llm.utils.word_util import (
    batch_remove_padding_eos,
    match_stop_words,
    remove_padding_eos_with_numpy,
    truncate_response_with_stop_words,
    truncate_token_with_stop_word_id,
)


@dataclass
class DecodeContext:
    """Context for decode step: config and precomputed stop-word slices."""

    generate_config: GenerateConfig
    stop_word_str_list: List[str]
    stop_word_str_slices: List[str]
    stop_word_ids: List[List[int]]
    stop_word_id_slices: List[List[int]]
    eos_token_id: int


def process_stop_id(
    generate_config: GenerateConfig,
    generate_output: GenerateOutput,
    tokens: Any,
    stop_word_ids: List[List[int]],
    stop_word_id_slices: List[List[int]],
) -> Any:
    """Truncate token list at stop-word id (or slices); returns tokens for decode."""
    if not generate_config.print_stop_words:
        if not generate_output.finished:
            tokens = truncate_token_with_stop_word_id(tokens, stop_word_id_slices)
        else:
            tokens = truncate_token_with_stop_word_id(tokens, stop_word_ids)
    return tokens


def process_stop_str(
    generate_config: GenerateConfig,
    generate_output: GenerateOutput,
    text: str,
    all_text: str,
    stop_word_str_list: List[str],
    stop_word_str_slices: List[str],
    token_buffer: str,
) -> Tuple[str, str]:
    """Apply stop-word string matching and truncation; returns (text, token_buffer)."""
    if generate_config.return_incremental:
        text = token_buffer + text

    if stop_word_str_list:
        stop_idx, stop_len = match_stop_words(text, stop_word_str_list)
        if stop_idx != -1:
            if not generate_config.print_stop_words:
                text = text[:stop_idx]
            else:
                text = text[: stop_idx + stop_len]
            token_buffer = ""
            generate_output.finished = True

    if generate_output.finished:
        return text, token_buffer

    if generate_config.return_incremental or not generate_config.print_stop_words:
        trunc_text = truncate_response_with_stop_words(
            text, stop_word_str_slices, generate_config.is_streaming, True
        )
        if generate_config.return_incremental:
            token_buffer = text[len(trunc_text) :]
        text = trunc_text

    return text, token_buffer


def _tokens_from_tensor(
    token_tensor: torch.Tensor,
    ignore_eos: bool,
    eos_token_id: int,
) -> Any:
    """Flatten token tensor to 1D and optionally remove EOS padding. Returns array-like for decode."""
    tokens_np = token_tensor.cpu().numpy().flatten()
    if not ignore_eos:
        return remove_padding_eos_with_numpy(tokens_np, eos_token_id)
    return tokens_np.reshape(-1)


def _collect_beam_token_lists(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
) -> List[Any]:
    """Collect token lists from beam outputs: concat output_ids, remove EOS, return list per beam."""
    generate_config = ctx.generate_config
    all_output_ids = torch.cat(
        [go.output_ids for go in generate_outputs.generate_outputs], dim=0
    )
    all_output_ids_np = all_output_ids.cpu().numpy()
    if not generate_config.ignore_eos:
        processed_tokens_np_list = batch_remove_padding_eos(
            all_output_ids_np, ctx.eos_token_id
        )
        return [tokens.tolist() for tokens in processed_tokens_np_list]
    return all_output_ids_np.tolist()


def _collect_single_sequence_token_lists(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
    ouput_tokens_list: List[torch.Tensor],
) -> Tuple[List[Any], List[torch.Tensor]]:
    """Collect token lists per sequence: accumulate output_ids, remove EOS, return (tokens_lists, ouput_tokens_list)."""
    generate_config = ctx.generate_config
    if len(ouput_tokens_list) == 0:
        ouput_tokens_list = [
            torch.empty(0, dtype=torch.int32)
            for _ in range(len(generate_outputs.generate_outputs))
        ]
    tokens_lists_for_decode_input: List[Any] = []
    for i, generate_output in enumerate(generate_outputs.generate_outputs):
        if len(ouput_tokens_list[i]) == 0:
            ouput_tokens_list[i] = generate_output.output_ids
        else:
            ouput_tokens_list[i] = torch.cat(
                (ouput_tokens_list[i], generate_output.output_ids), dim=1
            )
            generate_output.output_ids = ouput_tokens_list[i]
        tokens = _tokens_from_tensor(
            generate_output.output_ids,
            generate_config.ignore_eos,
            ctx.eos_token_id,
        )
        tokens_lists_for_decode_input.append(tokens)
    return tokens_lists_for_decode_input, ouput_tokens_list


def _collect_token_lists_non_incremental(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
    ouput_tokens_list: List[torch.Tensor],
) -> Tuple[List[Any], List[torch.Tensor]]:
    """Collect raw token lists for decode (EOS already removed). Beam vs single-sequence branch."""
    if ctx.generate_config.has_num_beams():
        return _collect_beam_token_lists(ctx, generate_outputs), ouput_tokens_list
    return _collect_single_sequence_token_lists(
        ctx, generate_outputs, ouput_tokens_list
    )


def _apply_stop_id_and_build_decode_lists(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
    tokens_lists_for_decode_input: List[Any],
) -> Tuple[List[int], List[Any]]:
    """Apply stop_id truncation per sequence and build lists for batch_decode."""
    output_lens: List[int] = []
    token_lists_to_decode: List[Any] = []
    generate_config = ctx.generate_config

    for i, generate_output in enumerate(generate_outputs.generate_outputs):
        tokens_list = tokens_lists_for_decode_input[i]
        output_lens.append(len(tokens_list))
        processed_tokens = process_stop_id(
            generate_config,
            generate_output,
            tokens_list,
            ctx.stop_word_ids,
            ctx.stop_word_id_slices,
        )
        token_lists_to_decode.append(processed_tokens)

    return output_lens, token_lists_to_decode


def _batch_decode_texts(
    ctx: DecodeContext,
    tokenizer: BaseTokenizer,
    token_lists_to_decode: List[Any],
) -> Tuple[List[str], List[str]]:
    """Batch decode token lists and strip replacement char."""
    generate_config = ctx.generate_config
    decoded_batch = tokenizer.batch_decode(
        token_lists_to_decode,
        skip_special_tokens=generate_config.skip_special_tokens,
    )
    newly_decoded_texts = [text.rstrip("\uFFFD") for text in decoded_batch]
    all_texts = newly_decoded_texts
    return newly_decoded_texts, all_texts


def _apply_stop_str_and_prefix(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
    newly_decoded_texts: List[str],
    all_texts: List[str],
    token_buffers: Optional[List[str]] = None,
) -> List[str]:
    """Apply stop_str per text and out_prefix. If token_buffers is not None, write back buffers."""
    generate_config = ctx.generate_config
    n = len(all_texts)
    if token_buffers is None:
        token_buffers = [""] * n
    final_texts: List[str] = []
    for i in range(n):
        processed_text, token_buffers[i] = process_stop_str(
            generate_config,
            generate_outputs.generate_outputs[i],
            newly_decoded_texts[i],
            all_texts[i],
            ctx.stop_word_str_list,
            ctx.stop_word_str_slices,
            token_buffers[i],
        )
        if generate_config.out_prefix:
            processed_text = generate_config.out_prefix + processed_text
        final_texts.append(processed_text)
    return final_texts


def _ensure_incremental_buffers(
    num_outputs: int,
    decoding_states: List[DecodingState],
    token_buffers: List[str],
    ouput_tokens_list: List[torch.Tensor],
) -> Tuple[List[DecodingState], List[str], List[torch.Tensor]]:
    """Initialize decoding_states, token_buffers, ouput_tokens_list if empty."""
    if len(token_buffers) == 0:
        token_buffers = [""] * num_outputs
    if len(decoding_states) == 0:
        decoding_states = [DecodingState() for _ in range(num_outputs)]
    if len(ouput_tokens_list) == 0:
        ouput_tokens_list = [
            torch.empty(0, dtype=torch.int32) for _ in range(num_outputs)
        ]
    return decoding_states, token_buffers, ouput_tokens_list


def _one_incremental_step(
    i: int,
    ctx: DecodeContext,
    generate_output: GenerateOutput,
    ouput_tokens_list: List[torch.Tensor],
    decoding_states: List[DecodingState],
    tokenizer: BaseTokenizer,
) -> Tuple[int, str, str]:
    """Run one incremental decode step for output at index i. Returns (output_len, text_to_return, all_text)."""
    generate_config = ctx.generate_config
    ouput_tokens_list[i] = torch.cat(
        (ouput_tokens_list[i], generate_output.output_ids), dim=1
    )
    full_tokens_tensor = ouput_tokens_list[i]
    tokens = _tokens_from_tensor(
        full_tokens_tensor,
        generate_config.ignore_eos,
        ctx.eos_token_id,
    )
    tokens_list = tokens.tolist()
    output_len = len(tokens_list)
    processed_tokens = process_stop_id(
        generate_config,
        generate_output,
        tokens_list,
        ctx.stop_word_ids,
        ctx.stop_word_id_slices,
    )
    new_text = IncrementDecodingUtils.detokenize_incrementally(
        tokenizer, processed_tokens, decoding_states[i]
    )
    decoding_states[i].all_text += new_text
    text_to_return = (
        new_text if generate_config.return_incremental else decoding_states[i].all_text
    )
    return output_len, text_to_return, decoding_states[i].all_text


def _run_incremental_steps(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
    ouput_tokens_list: List[torch.Tensor],
    decoding_states: List[DecodingState],
    tokenizer: BaseTokenizer,
) -> Tuple[List[int], List[str], List[str]]:
    """Run one incremental decode step per output; return output_lens, newly_decoded_texts, all_texts."""
    output_lens: List[int] = []
    newly_decoded_texts: List[str] = []
    all_texts: List[str] = []
    for i, generate_output in enumerate(generate_outputs.generate_outputs):
        out_len, text_to_return, all_text = _one_incremental_step(
            i, ctx, generate_output, ouput_tokens_list, decoding_states, tokenizer
        )
        output_lens.append(out_len)
        newly_decoded_texts.append(text_to_return)
        all_texts.append(all_text)
    return output_lens, newly_decoded_texts, all_texts


def decode_non_incremental_tokens(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
    ouput_tokens_list: List[torch.Tensor],
    tokenizer: BaseTokenizer,
) -> Tuple[List[str], List[int], List[torch.Tensor]]:
    """Decode full sequences (beam or non-streaming). Returns (final_texts, output_lens, ouput_tokens_list)."""
    tokens_lists_for_decode_input, ouput_tokens_list = (
        _collect_token_lists_non_incremental(ctx, generate_outputs, ouput_tokens_list)
    )
    output_lens, token_lists_to_decode = _apply_stop_id_and_build_decode_lists(
        ctx, generate_outputs, tokens_lists_for_decode_input
    )
    newly_decoded_texts, all_texts = _batch_decode_texts(
        ctx, tokenizer, token_lists_to_decode
    )
    final_texts = _apply_stop_str_and_prefix(
        ctx, generate_outputs, newly_decoded_texts, all_texts, token_buffers=None
    )
    return (final_texts, output_lens, ouput_tokens_list)


def decode_incremental_tokens(
    ctx: DecodeContext,
    generate_outputs: GenerateOutputs,
    decoding_states: List[DecodingState],
    token_buffers: List[str],
    ouput_tokens_list: List[torch.Tensor],
    tokenizer: BaseTokenizer,
) -> Tuple[
    List[str],
    List[int],
    List[DecodingState],
    List[str],
    List[torch.Tensor],
]:
    """Decode streaming step incrementally. Returns (final_texts, output_lens, decoding_states, token_buffers, ouput_tokens_list)."""
    num_outputs = len(generate_outputs.generate_outputs)
    decoding_states, token_buffers, ouput_tokens_list = _ensure_incremental_buffers(
        num_outputs, decoding_states, token_buffers, ouput_tokens_list
    )
    output_lens, newly_decoded_texts, all_texts = _run_incremental_steps(
        ctx, generate_outputs, ouput_tokens_list, decoding_states, tokenizer
    )
    final_texts = _apply_stop_str_and_prefix(
        ctx,
        generate_outputs,
        newly_decoded_texts,
        all_texts,
        token_buffers=token_buffers,
    )
    return (
        final_texts,
        output_lens,
        decoding_states,
        token_buffers,
        ouput_tokens_list,
    )
