from multiprocessing import Lock, Value


class RequestCounter:
    """仅用于生成递增的 request sequence 号，不做任何限流"""

    def __init__(self):
        self._counter = Value("i", 0)
        self._lock = Lock()

    def next_sequence(self) -> int:
        with self._lock:
            self._counter.value += 1
            return self._counter.value
