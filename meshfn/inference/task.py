from dataclasses import dataclass
from typing import Hashable, Tuple, Any


@dataclass
class StreamingTaskEntry:
    uids: Tuple[Hashable, ...]
    index: Tuple[int, ...]
    batch: Any
    finished: bool


@dataclass
class TaskEntry:
    uids: Tuple[Hashable, ...]
    batch: Any
