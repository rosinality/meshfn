from typing import Dict
from queue import Queue, Empty
from threading import Lock
from typing import Any
import time

from torch.distributed import rpc

from meshfn.inference.threading import use_lock


def rpc_queue_can_put(q: rpc.RRef) -> bool:
    q = q.local_value()

    return not q.full()


def rpc_queue_put(q: rpc.RRef, data: any) -> None:
    q = q.local_value()
    q.put(data)


class RPCPipe:
    queues: Dict[str, Queue] = {}
    lock = Lock()

    def __init__(self, name: str, src: str, dst: str, max_size: int = 0) -> None:
        self.rpc_info = rpc.get_worker_info()
        self.name = name
        self.src = src
        self.dst = dst
        self.remote_queue: rpc.RRef = None
        self.local_queue: Queue[Any] = None

        with use_lock(self.lock):
            if src == self.rpc_info.name:
                if name in self.queues:
                    raise ValueError(
                        f"pipe '{name}' already exists on {self.rpc_info.name}"
                    )

                self.remote_queue = self.get_remote_queue(max_size)
                self.queues[name] = self.remote_queue

    @classmethod
    def rpc_create_local_queue(cls, name: str, max_size: int) -> Queue:
        with use_lock(cls.lock):
            if name in cls.queues:
                raise ValueError(f"pipe '{name}' already exists")

            cls.queues[name] = Queue(max_size)

            return cls.queues[name]

    def get_remote_queue(self, max_size: int) -> rpc.RRef:
        return rpc.remote(
            self.dst, self.rpc_create_local_queue, args=(self.name, max_size)
        )

    def prepare_local_queue(self) -> None:
        if self.local_queue is not None:
            return

        with use_lock(self.lock):
            if self.name in self.queues:
                self.local_queue = self.queues[self.name]

    def recv(self) -> Any:
        if self.dst != self.rpc_info.name:
            raise ValueError(
                f"destination '{self.dst}' is different from {self.rpc_info.name}"
            )

        while True:
            self.prepare_local_queue()

            if self.local_queue is not None:
                return self.local_queue.get()

            time.sleep(0.01)

    def recv_nowait(self) -> Any:
        if self.dst != self.rpc_info.name:
            raise ValueError(
                f"destination '{self.dst}' is different from {self.rpc_info.name}"
            )

        self.prepare_local_queue()

        if self.local_queue is not None:
            try:
                return self.local_queue.get_nowait()

            except Empty:
                raise RuntimeError("pipe is empty")

        raise RuntimeError("local queue is not created")

    def send(self, data: Any) -> None:
        if self.src != self.rpc_info.name:
            raise ValueError(
                f"destination '{self.dst}' is different from {self.rpc_info.name}"
            )

        while not rpc.rpc_sync(self.dst, rpc_queue_can_put, args=(self.remote_queue,)):
            time.sleep(0.1)

        rpc.rpc_sync(self.dst, rpc_queue_put, args=(self.remote_queue, data))
