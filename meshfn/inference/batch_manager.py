from dataclasses import dataclass
from typing import Any, Deque, Hashable, Tuple, List, Iterable

import torch

from meshfn.inference.task import TaskEntry, StreamingTaskEntry


@dataclass
class SubmitEntry:
    uid: Hashable
    data: Any


class BatchManager:
    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()

        return TaskEntry((entry.uid,), entry.data), {}

    def split_batch(
        self, task_entry: TaskEntry, **kwargs: Any
    ) -> Iterable[Tuple[Hashable, Any]]:
        return [(task_entry.uids[0], 0, task_entry.batch)]


class BatchManagerForHFGeneration(BatchManager):
    def __init__(self, max_batch_size: int = 1, pad_token_id: int = 0) -> None:
        super().__init__()

        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    def left_padding(self, batch_inputs):
        max_len = max(len(inputs["input_ids"]) for inputs in batch_inputs)
        outputs = {"input_ids": []}

        for inputs in batch_inputs:
            input_ids = inputs["input_ids"]
            padding_len = max_len - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            outputs["input_ids"].append(input_ids)

        for k in outputs:
            outputs[k] = torch.tensor(outputs[k])

        return outputs, max_len

    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        batch = [entry.data]

        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break

            if self.batch_attributes(entry) != self.batch_attributed(q[0]):
                break

            if q[0].data["max_tokens"] > entry.data["max_tokens"]:
                break

            e = q.popleft()

            batch.append(e.data)
            uids.append(e.uid)

        inputs, max_len = self.left_padding(batch)
        trunc_lens = []

        for data in batch:
            trunc_lens.append(max_len + data["max_tokens"])

        inputs["top_p"] = entry.data["top_p"]
        inputs["temperature"] = entry.data["temperature"]
        inputs["max_new_tokens"] = entry.data["max_tokens"] - max_len

        return TaskEntry(tuple(uids), inputs), {}

    def split_batch(
        self, task_entry: StreamingTaskEntry
    ) -> List[Tuple[Hashable, int, Any]]:
        results = []

        for uid, index, *output in zip(
            task_entry.uids, task_entry.index, *task_entry.batch
        ):
            results.append((uid, index, output))

        return results
