from typing import *
import threading


class Atomic:
    class AtomicContextManager:
        def __init__(self, atomic: 'Atomic'):
            self._atomic = atomic
            self._exited = False

        def __enter__(self):
            self._atomic._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._exited = True
            self._atomic._lock.release()

        def _ensure_not_exited(self) -> None:
            if self._exited:
                raise RuntimeError('Trying to access AtomicContextManager after its exit.')

        def value(self) -> Any:
            self._ensure_not_exited()
            return self._atomic._val

        def set(self, value: Any) -> None:
            self._ensure_not_exited()
            self._atomic._val = value

    def __init__(self, value: Any):
        self._val = value
        self._lock = threading.Lock()

    def lock(self) -> AtomicContextManager:
        return self.__class__.AtomicContextManager(self)

    def value(self) -> Any:
        with self.lock() as val:
            return val.value()
