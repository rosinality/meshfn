from typing import Dict, List, Optional

from meshfn.distributed import ParallelMode, ParallelContext
from meshfn.nn.parallel import tensor1d


class Shard:
    def __init__(self, parallel_mode: ParallelMode):
        self.parallel_mode = parallel_mode

    def n_shard(self, parallel_context: ParallelContext):
        return parallel_context.world_size(self.parallel_mode)

    def place(self, tensor, dim: int, parallel_context: ParallelContext):
        world_size = parallel_context.world_size(self.parallel_mode)

        return tensor.chunk(world_size, dim=dim)


class Strategy:
    def __init__(
        self,
        placement: Dict[str, List[Optional[Shard]]],
        optional: bool = False,
        orig_name: Optional[str] = None,
    ):
        self.placement = placement
        self.optional = optional
        self.orig_name = orig_name


class LinearColumn1D:
    layer = tensor1d.LinearColumn1D
    strategy = {
        "weight": Strategy([Shard(ParallelMode.TENSOR_1D), None]),
        "bias": Strategy([Shard(ParallelMode.TENSOR_1D)], optional=True),
    }


class LinearRow1D:
    layer = tensor1d.LinearRow1D
    strategy = {
        "weight": Strategy([None, Shard(ParallelMode.TENSOR_1D)]),
        "bias": Strategy([None], optional=True),
    }


class VocabParallelEmbedding:
    layer = tensor1d.VocabParallelEmbedding
    strategy = {"weight": Strategy([Shard(ParallelMode.TENSOR_1D), None])}
