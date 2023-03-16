from contextlib import contextmanager
import signal
from threading import Lock


@contextmanager
def use_lock(lock: Lock):
    try:
        lock.acquire()

        yield

    finally:
        lock.release()


class Terminator:
    lock = Lock()
    called: bool = False

    @classmethod
    def shield(cls):
        with use_lock(cls.lock):
            cls.called = True

    @classmethod
    def terminate(cls):
        with use_lock(cls.lock):
            if not cls.called:
                cls.called = True
                signal.raise_signal(signal.SIGINT)
