from typing import *
import threading


class LockContextManager:
    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._exited = False

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exited = True
        self._lock.release()

    def exited(self) -> bool:
        return self._exited


class Atomic:
    class AtomicContextManager(LockContextManager):
        def __init__(self, atomic: 'Atomic'):
            super().__init__(atomic._lock)
            self._atomic = atomic
            self._exited = False

        def __enter__(self):
            super().__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            super().__exit__(exc_type, exc_val, exc_tb)

        def value(self) -> Any:
            if self.exited():
                raise RuntimeError('Trying to access AtomicContextManager after its exit.')
            return self._atomic._val

        def set(self, value: Any) -> None:
            if self.exited():
                raise RuntimeError('Trying to access AtomicContextManager after its exit.')
            self._atomic._val = value

    def __init__(self, value: Any):
        self._val = value
        self._lock = threading.Lock()

    def lock(self) -> AtomicContextManager:
        return self.__class__.AtomicContextManager(self)

    def value(self) -> Any:
        with self.lock() as val:
            return val.value()
