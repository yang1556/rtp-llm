"""Base endpoint with common request pipeline for frontend worker and OpenAI endpoint."""

import asyncio
import json
from typing import Any, Callable, Dict, Optional, Tuple, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.server.misc import format_exception
from rtp_llm.structure.request_extractor import RequestExtractor
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    ConcurrencyException,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter


class BaseEndpoint:
    """Base class for all endpoints with common pipeline: _check_request + inference_request + handle_request."""

    def __init__(
        self,
        global_controller: ConcurrencyController,
        access_logger: AccessLogger,
        rank_id: str = "0",
        server_id: str = "0",
        frontend_worker=None,
        active_requests: Optional[AtomicCounter] = None,
    ):
        self._global_controller = global_controller
        self._access_logger = access_logger
        self.rank_id = rank_id
        self.server_id = server_id
        self._frontend_worker = frontend_worker
        self._active_requests = (
            active_requests if active_requests is not None else AtomicCounter()
        )

    def _log_and_report_exception(
        self, request: Dict[str, Any], e: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Log exception to access_logger and report CANCEL_QPS or ERROR_QPS. Returns format_exception(e) for non-CancelledError, None for CancelledError."""
        self._access_logger.log_exception_access(request, e)
        source = request.get("source", "unknown")
        tags = {
            "rank_id": self.rank_id,
            "server_id": self.server_id,
            "source": source,
        }
        if isinstance(e, asyncio.CancelledError):
            kmonitor.report(AccMetrics.CANCEL_QPS_METRIC, 1, tags)
            return None
        format_e = format_exception(e)
        kmonitor.report(
            AccMetrics.ERROR_QPS_METRIC,
            1,
            {**tags, "error_code": str(format_e.get("error_code_str", -1))},
        )
        return format_e

    def _handle_exception(
        self, request: Union[Dict[str, Any], Any], e: BaseException
    ) -> ORJSONResponse:
        """Handle exceptions and return proper error response (no stack trace to client)."""
        request_dict = (
            request
            if isinstance(request, dict)
            else (
                request.model_dump(exclude_none=True)
                if hasattr(request, "model_dump")
                else {}
            )
        )
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
            exception_json = format_exception(e)
        else:
            exception_json = self._log_and_report_exception(request_dict, e)
            if exception_json is None:
                exception_json = format_exception(e)
        return ORJSONResponse(exception_json, status_code=500)

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        """Collect complete response and log access."""
        complete_response = await res.gen_complete_response_once()
        complete_response = (
            complete_response.model_dump(exclude_none=True)
            if isinstance(complete_response, BaseModel)
            else complete_response
        )
        self._access_logger.log_success_access(req, complete_response)
        return complete_response

    async def stream_response(
        self,
        request: Dict[str, Any],
        response_generator: CompleteResponseAsyncGenerator,
    ):
        """Generate streaming response with metrics reported during iteration."""
        response_data_prefix = "data: " if request.get("stream", False) else "data:"
        async for chunk in self._stream_with_controller_lifecycle(
            request, response_generator, response_data_prefix
        ):
            yield chunk

    @staticmethod
    def _step_output_len_from_response(response: Any) -> int:
        """Extract step_output_len from response.aux_info for iter RT metric."""
        if not hasattr(response, "aux_info"):
            return 1
        aux = response.aux_info
        if isinstance(aux, list):
            return sum(info.get("step_output_len", 1) for info in aux) or 1
        if isinstance(aux, dict):
            return max(aux.get("step_output_len", 1), 1)
        return 1

    def _report_stream_iter_metrics(
        self,
        response: Any,
        end_time: float,
        last_iterate_time: float,
        first_token: bool,
    ) -> Tuple[float, bool]:
        """Report first-token or iter RT and ITER_QPS; return (end_time, first_token after)."""
        rt = end_time - last_iterate_time
        if first_token:
            kmonitor.report(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC, rt)
            first_token = False
        else:
            step_len = self._step_output_len_from_response(response)
            kmonitor.report(GaugeMetrics.RESPONSE_ITER_RT_METRIC, rt / step_len)
        kmonitor.report(
            AccMetrics.ITER_QPS_METRIC,
            1,
            {"rank_id": self.rank_id, "server_id": self.server_id},
        )
        return (end_time, first_token)

    async def _stream_with_controller_lifecycle(
        self,
        request: Dict[str, Any],
        response_generator: CompleteResponseAsyncGenerator,
        response_data_prefix: str,
    ):
        """Stream body with try/except/finally: log+report on exception, decrement in finally."""
        is_openai_response = response_data_prefix == "data: "
        start_time = current_time_ms()
        last_iterate_time = start_time
        first_token = True
        iter_count = 0
        iter_tags = {"rank_id": self.rank_id, "server_id": self.server_id}
        try:
            async for res in response_generator:
                end_time = current_time_ms()
                last_iterate_time, first_token = self._report_stream_iter_metrics(
                    res, end_time, last_iterate_time, first_token
                )
                iter_count += 1
                yield response_data_prefix + res.model_dump_json(
                    exclude_none=True
                ) + "\r\n\r\n"
                await asyncio.sleep(0)
            kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time
            )
            kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, iter_tags)
            if not is_openai_response:
                yield "data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(
                request, response_generator
            )
        except asyncio.CancelledError as e:
            self._log_and_report_exception(request, e)
        except BaseException as e:
            format_e = self._log_and_report_exception(request, e)
            if format_e is not None:
                yield response_data_prefix + json.dumps(
                    format_e, ensure_ascii=False
                ) + "\r\n\r\n"
        finally:
            self._global_controller.decrement()

    def _check_is_streaming(self, request_dict: Dict[str, Any]) -> bool:
        """Determine if request should use streaming mode."""
        if self._frontend_worker is not None and hasattr(
            self._frontend_worker, "is_streaming"
        ):
            return self._frontend_worker.is_streaming(request_dict)
        return RequestExtractor.is_streaming(request_dict) or request_dict.get(
            "stream", False
        )

    # ---------- Abstract: subclass implements ----------

    def _check_request(self, request: Any, req_id: int) -> Dict[str, Any]:
        """Parse/validate request and set request_id in dict. Subclass must implement."""
        raise NotImplementedError("Subclass must implement _check_request")

    def inference_request(
        self,
        request_dict: Dict[str, Any],
        raw_request: Optional[Request] = None,
    ) -> CompleteResponseAsyncGenerator:
        """Build response generator from request_dict. Subclass must implement (for streaming path)."""
        raise NotImplementedError("Subclass must implement inference_request")

    def _finish_with_response(
        self,
        request_dict: Dict[str, Any],
        complete_response: Any,
        logable_for_log: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ORJSONResponse:
        """Decrement, log success access, return ORJSONResponse. Shared by direct and non-streaming paths."""
        self._global_controller.decrement()
        self._access_logger.log_success_access(
            request_dict,
            logable_for_log if logable_for_log is not None else complete_response,
        )
        return ORJSONResponse(
            content=complete_response, headers=headers if headers else {}
        )

    # ---------- Unified pipeline ----------

    async def _with_active_requests(
        self,
        request: Any,
        raw_request: Request,
        impl: Callable[
            [Any, Request],
            Any,
        ],
    ) -> Union[StreamingResponse, ORJSONResponse]:
        """Increment active_requests, run impl, decrement in finally."""
        self._active_requests.increment()
        try:
            return await impl(request, raw_request)
        finally:
            self._active_requests.decrement()

    def _convert_to_dict(self, request: Any) -> Dict[str, Any]:
        """Convert request to dict for error handling."""
        if isinstance(request, dict):
            return request
        if hasattr(request, "model_dump"):
            return request.model_dump(exclude_none=True)
        if hasattr(request, "dict"):
            return request.dict(exclude_none=True)
        return {}

    async def handle_request(
        self, request: Any, raw_request: Request
    ) -> Union[StreamingResponse, ORJSONResponse]:
        """Unified pipeline: active_requests, increment, _check_request, metrics, inference_request, stream or collect."""
        return await self._with_active_requests(
            request, raw_request, self._handle_request_impl
        )

    async def _obtain_request_dict_and_validate(
        self, request: Any, raw_request: Request
    ) -> Union[Dict[str, Any], ORJSONResponse]:
        """Increment controller, check_request, report QPS/log, disconnect check. Returns request_dict or ORJSONResponse on check error. Caller must decrement on exception."""
        req_id = self._global_controller.increment()
        try:
            request_dict = self._check_request(request, req_id)
        except Exception as e:
            self._global_controller.decrement()
            return self._handle_exception(request, e)
        kmonitor.report(
            AccMetrics.QPS_METRIC,
            1,
            {
                "rank_id": self.rank_id,
                "server_id": self.server_id,
                "source": request_dict.get("source", "unknown"),
            },
        )
        self._access_logger.log_query_access(request_dict)
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        return request_dict

    async def _handle_request_impl(
        self, request: Any, raw_request: Request
    ) -> Union[StreamingResponse, ORJSONResponse]:
        """Controller lifecycle + core: obtain request_dict, then direct or generator path."""
        try:
            result = await self._obtain_request_dict_and_validate(request, raw_request)
            if isinstance(result, ORJSONResponse):
                return result
            request_dict = result

            response_generator = self.inference_request(request_dict, raw_request)

            if self._check_is_streaming(request_dict):
                return StreamingResponse(
                    self.stream_response(request_dict, response_generator),
                    media_type="text/event-stream",
                )

            # Non-streaming: iterate with report, then collect and finish
            start_time = current_time_ms()
            last_iterate_time = start_time
            first_token = True
            iter_count = 0
            iter_tags = {"rank_id": self.rank_id, "server_id": self.server_id}
            async for response in response_generator:
                if await raw_request.is_disconnected():
                    await response_generator.aclose()
                    raise asyncio.CancelledError("client disconnects")
                end_time = current_time_ms()
                last_iterate_time, first_token = self._report_stream_iter_metrics(
                    response, end_time, last_iterate_time, first_token
                )
                iter_count += 1
            kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time
            )
            kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, iter_tags)
            complete_response = await response_generator.gen_complete_response_once()
            complete_response = (
                complete_response.model_dump(exclude_none=True)
                if isinstance(complete_response, BaseModel)
                else complete_response
            )
            return self._finish_with_response(
                request_dict,
                complete_response,
                logable_for_log=complete_response,
            )
        except BaseException as e:
            self._global_controller.decrement()
            req_for_error = (
                request_dict
                if "request_dict" in locals()
                else self._convert_to_dict(request)
            )
            return self._handle_exception(req_for_error, e)
