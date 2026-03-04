"""Bridge to run an async generator in a dedicated thread and yield results synchronously."""

import asyncio
import queue
import threading
from typing import AsyncGenerator, Callable, Iterator, TypeVar

T = TypeVar("T")


def async_iterator_to_sync(
    async_gen_factory: Callable[[], AsyncGenerator[T, None]],
) -> Iterator[T]:
    """Run the async generator produced by async_gen_factory in a dedicated thread;
    yield each item to the caller. Exceptions are propagated; the async generator
    is closed in finally when the loop ends or on error.
    """
    q: queue.Queue = queue.Queue()

    async def generator():
        res = None
        try:
            res = async_gen_factory()
            async for x in res:
                q.put(x)
            q.put(None)
        except Exception as e:
            q.put(e)
        finally:
            if res is not None:
                res.aclose()

    def start_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generator())

    background_thread = threading.Thread(target=start_loop)
    background_thread.start()
    try:
        while True:
            try:
                r = q.get(timeout=0.01)
                if r is None:
                    break
                if isinstance(r, Exception):
                    raise r
                yield r
            except queue.Empty:
                continue
    finally:
        background_thread.join()
