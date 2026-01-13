import asyncio
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import grpc.aio
import torch

# Assuming these imports are correct relative to your project structure
from .proto.model_rpc_service_pb2 import (
    EmptyPB,
    GenerateInputPB,
    GenerateOutputsPB,
    TensorPB,
)
from .proto.model_rpc_service_pb2_grpc import RpcServiceStub
from .scheduler import Estimator, EstimatorSample


@dataclass
class ServerStatus:
    server_running_batchs: int
    server_waiting_batchs: int
    free_blocks: int
    total_blocks: int
    step: int


@dataclass
class AuxInfo:
    """Data class to hold Auxiliary Information from the server."""

    cost_time_us: int = 0
    iter_count: int = 0
    input_len: int = 0
    output_len: int = 0
    prefix_len: int = 0
    total_reuse_len: int = 0
    local_reuse_len: int = 0
    remote_reuse_len: int = 0
    fallback_tokens: int = 0
    fallback_times: int = 0
    wait_time_us: int = 0


@dataclass
class LLMRequest:
    """Data class to hold Large Language Model (LLM) request parameters."""

    tokens: list[int]
    rmp_tokens: list[int]  # tokens after remove padding
    # extra field from verl
    attention_mask: list
    position_ids: list

    request_id: int = 0
    parent_id: int = 0
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 1000
    n: int = 1
    timestamp: float = 0.0  # Timestamp when the request was sent
    launch_step: int = 0  # for scheduler

    def __hash__(self):
        # Use request_id for a unique hash, which is more reliable than tuple(tokens)
        return self.request_id


@dataclass
class LLMResponse:
    """Data class to hold Large Language Model (LLM) response parameters."""

    tokens: torch.Tensor
    logits: torch.Tensor
    timestamp: float
    error: Optional[str] = None
    status_code: int = 0  # For gRPC, this might represent a gRPC status code, 0 for OK
    aux_info: Optional[AuxInfo] = None


@dataclass
class RequestLog:
    """Data class to log each request and its corresponding response."""

    request: LLMRequest
    response: LLMResponse


def _convert_tensor_pb_to_torch(tensor_pb: TensorPB) -> torch.Tensor:
    """
    Converts a TensorPB (protobuf) object to a torch.Tensor.
    """
    if not (len(tensor_pb.shape) > 0 and tensor_pb.shape[0] > 0):
        return torch.tensor([], dtype=torch.float32)

    if tensor_pb.data_type == TensorPB.DataType.FP32:
        return torch.frombuffer(tensor_pb.fp32_data, dtype=torch.float32).reshape(
            list(tensor_pb.shape)
        )
    elif tensor_pb.data_type == TensorPB.DataType.INT32:
        return torch.frombuffer(tensor_pb.int32_data, dtype=torch.int32).reshape(
            list(tensor_pb.shape)
        )
    elif tensor_pb.data_type == TensorPB.DataType.FP16:
        return torch.frombuffer(tensor_pb.fp16_data, dtype=torch.float16).reshape(
            list(tensor_pb.shape)
        )
    elif tensor_pb.data_type == TensorPB.DataType.BF16:
        return torch.frombuffer(tensor_pb.bf16_data, dtype=torch.bfloat16).reshape(
            list(tensor_pb.shape)
        )
    else:
        raise Exception("Unknown tensor data type")


def build_output_py(output_pb: GenerateOutputsPB) -> LLMResponse:
    """
    Builds an LLMResponse object from a GenerateOutputsPB (protobuf) object.
    """
    aux_info_data = None
    if output_pb.HasField("aux_info"):
        pb_aux = output_pb.aux_info
        aux_info_data = AuxInfo(
            cost_time_us=pb_aux.cost_time_us,
            iter_count=pb_aux.iter_count,
            input_len=pb_aux.input_len,
            output_len=pb_aux.output_len,
            prefix_len=pb_aux.prefix_len,
            total_reuse_len=pb_aux.total_reuse_len,
            local_reuse_len=pb_aux.local_reuse_len,
            remote_reuse_len=pb_aux.remote_reuse_len,
            fallback_tokens=pb_aux.fallback_tokens,
            fallback_times=pb_aux.fallback_times,
            wait_time_us=pb_aux.wait_time_us,
        )
    return LLMResponse(
        tokens=_convert_tensor_pb_to_torch(output_pb.output_ids).flatten(),
        logits=_convert_tensor_pb_to_torch(output_pb.logits),
        timestamp=time.time(),
        error=None,
        status_code=0,
        aux_info=aux_info_data,
    )


def build_input_pb(request: LLMRequest) -> GenerateInputPB:
    """
    Builds the input protobuf for the gRPC request.
    """
    input_pb = GenerateInputPB()
    input_pb.request_id = request.request_id
    input_pb.parent_id = request.parent_id
    input_pb.token_ids.extend(request.rmp_tokens)
    generate_config_pb = input_pb.generate_config
    generate_config_pb.num_return_sequences = request.n
    generate_config_pb.max_new_tokens = request.max_new_tokens
    generate_config_pb.top_k = request.top_k
    generate_config_pb.top_p = request.top_p
    generate_config_pb.temperature = request.temperature
    generate_config_pb.repetition_penalty = 1.0
    generate_config_pb.num_beams = 1
    generate_config_pb.do_sample = True

    return input_pb


class AsyncGrpcClient:
    """gRPC client for asynchronous operations."""

    def __init__(
        self, target_url: str, concurrency_limit: int = 1536, send_interval: float = 1.0
    ):
        self.target_url = target_url
        self.running: bool = True
        self.request_logs: list[RequestLog] = []
        self.lock = threading.Lock()
        self.concurrency_limit = concurrency_limit

        self._worker_loop_event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_send_queue: Optional[asyncio.Queue] = None

        # **FIX**: Do NOT initialize channel/stub here. They must be created
        # within the worker thread's event loop.
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[RpcServiceStub] = None

        self._traffic_controller = Estimator()
        self._running_requests: set[LLMRequest] = set()

        self.status = ServerStatus(0, 0, 0, 0, 0)
        self.send_interval = send_interval

        self.GRACEFUL_SHUTDOWN_TIMEOUT = 60

        self.worker_init_event = threading.Event()

        self.worker = threading.Thread(target=self._run_worker_loop, daemon=True)
        self.worker.start()

        ready = self.worker_init_event.wait(timeout=5.0)  # Wait up to 5 seconds
        if not ready:
            raise RuntimeError("Worker thread failed to initialize in time.")

        # The check below is now more reliable.
        if self._worker_loop_event_loop is None or self._async_send_queue is None:
            raise RuntimeError("Failed to initialize worker event loop or queue.")

    def _run_worker_loop(self):
        """Helper method to run the asyncio event loop in the worker thread."""
        self._worker_loop_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop_event_loop)

        # Initialize the queue within the loop it will be used in.
        self._async_send_queue = asyncio.Queue()

        self._channel = grpc.aio.insecure_channel(self.target_url)
        self._stub = RpcServiceStub(self._channel)

        # Signal that initialization is complete
        self.worker_init_event.set()

        self._worker_loop_event_loop.run_until_complete(self._worker_loop())
        self._worker_loop_event_loop.close()

    async def _try_query_status(self):
        return
        """Asynchronously queries the server status."""
        try:
            recv = await self._stub.QueryServerStatus(EmptyPB())
            self.status = ServerStatus(
                free_blocks=recv.free_blocks,
                total_blocks=recv.total_blocks,
                server_running_batchs=recv.running_batchs,
                server_waiting_batchs=recv.waiting_batchs,
                step=recv.step,
            )
        except Exception as e:
            # print(f"Error querying server status: {e}")
            self.status = ServerStatus(0, 0, 0, 0, 0)

    async def _send_grpc_request(self, request: LLMRequest) -> list[RequestLog]:
        """
        Sends a gRPC request and processes the streaming response.
        The request is removed from `_running_requests` upon completion or error.
        """
        results: list[RequestLog] = []
        try:
            input_pb = build_input_pb(request)
            async for grpc_response_pb in self._stub.GenerateStreamCall(input_pb):
                for output in grpc_response_pb.generate_outputs:
                    if not output.finished:
                        continue
                    response = build_output_py(output)
                    results.append(RequestLog(request=request, response=response))
                    self._traffic_controller.update(len(response.tokens))

        except grpc.aio.AioRpcError as e:
            error_message = f"gRPC Error (code: {e.code().name}): {e.details()}"
            print(error_message, traceback.format_exc())
            results.append(
                RequestLog(
                    request,
                    LLMResponse(
                        torch.tensor([]),
                        torch.tensor([]),
                        time.time(),
                        error_message,
                        e.code().value,
                    ),
                )
            )
        except Exception as e:
            error_message = f"Unexpected error in gRPC call: {e}"
            print(error_message, traceback.format_exc())
            results.append(
                RequestLog(
                    request,
                    LLMResponse(
                        torch.tensor([]),
                        torch.tensor([]),
                        time.time(),
                        error_message,
                        grpc.StatusCode.UNKNOWN.value,
                    ),
                )
            )
        finally:
            self._running_requests.discard(request)
        return results

    def enqueue(self, request: LLMRequest) -> None:
        """Adds a request to the processing queue from any thread."""
        if (
            not self._worker_loop_event_loop
            or not self._worker_loop_event_loop.is_running()
        ):
            raise RuntimeError(
                "Worker event loop is not running. Cannot enqueue request."
            )
        request.timestamp = time.time()
        asyncio.run_coroutine_threadsafe(
            self._async_send_queue.put(request), self._worker_loop_event_loop
        )

    async def _worker_loop(self):
        """
        Main worker loop with the new logic: query, build, then process queue based on traffic controller.
        """
        while self.running:
            # 1. Query server status, update traffic controller's states.
            await self._try_query_status()
            # print(f"Server Status: Free Blocks={self.status.free_blocks}, Running={self.status.server_running_batchs}, Waiting={self.status.server_waiting_batchs}")

            # 2. Build traffic controller model based on currently running requests
            pass

            # 3. Process items from the queue based on the traffic controller's advice
            # We check the number of items once to avoid an infinite loop within one cycle.
            num_to_check = self._async_send_queue.qsize()
            for _ in range(num_to_check):
                # Check for available concurrency slots before getting an item
                if len(self._running_requests) >= self.concurrency_limit:
                    break

                try:
                    request = self._async_send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break  # Queue is empty, exit the loop

                # 4. Decide whether to send the request using the traffic controller
                estimated_fail_rate = 0.0

                # 5. Send if probability is high, otherwise re-queue and break
                if True or estimated_fail_rate < 0.1:
                    request.launch_step = self.status.step
                    self._running_requests.add(request)

                    async def process_request(req: LLMRequest):
                        records = await self._send_grpc_request(req)
                        with self.lock:
                            self.request_logs.extend(records)
                        self._async_send_queue.task_done()

                    asyncio.create_task(process_request(request))
                else:
                    print(
                        f"Acceptance prob <= 0.9. Re-queuing request {request.request_id} and pausing for this cycle."
                    )
                    # Re-queue the request and stop processing more items in this cycle
                    await self._async_send_queue.put(request)
                    break

            # Wait for the next interval before the next status query
            await asyncio.sleep(self.send_interval)
            # self._traffic_controller.plot_estimations(fname=self.status.step)

        # Cleanup routine when self.running becomes False
        await self._shutdown_cleanup()

    async def _shutdown_cleanup(self):
        """Handles graceful shutdown cleanup tasks within the event loop."""
        while not self._async_send_queue.empty():
            try:
                request = self._async_send_queue.get_nowait()
                with self.lock:
                    self.request_logs.append(
                        RequestLog(
                            request=request,
                            response=LLMResponse(
                                torch.tensor([]),
                                torch.tensor([]),
                                time.time(),
                                "Client shutting down, request not processed.",
                                grpc.StatusCode.UNAVAILABLE.value,
                            ),
                        )
                    )
                self._async_send_queue.task_done()
            except asyncio.QueueEmpty:
                break

    def collect(self) -> list[RequestLog]:
        """Collects all completed request logs and clears the internal list."""
        with self.lock:
            logs = self.request_logs.copy()
            self.request_logs.clear()
        return logs

    def close(self):
        """Closes the client, worker thread, and gRPC channel."""
        self.running = False

        if self._worker_loop_event_loop and self._worker_loop_event_loop.is_running():
            # Schedule channel closing on the worker's event loop
            future = asyncio.run_coroutine_threadsafe(
                self._channel.close(), self._worker_loop_event_loop
            )
            try:
                future.result(timeout=5)
                print("gRPC channel closed successfully.")
            except (asyncio.TimeoutError, Exception) as e:
                print(f"Warning: Error closing gRPC channel: {e}")

        if self.worker.is_alive():
            print(
                f"Waiting for worker thread to terminate (timeout: {self.GRACEFUL_SHUTDOWN_TIMEOUT}s)..."
            )
            self.worker.join(timeout=self.GRACEFUL_SHUTDOWN_TIMEOUT)

        if self.worker.is_alive():
            print("Warning: Worker thread did not terminate gracefully.")
        else:
            print("Worker thread terminated.")
