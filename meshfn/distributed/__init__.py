from meshfn.distributed.parallel_context import ParallelContext
from meshfn.distributed.parallel_mode import ParallelMode
from meshfn.distributed.seed.seed_manager import SEEDS
from meshfn.distributed.collectives import broadcast, broadcast_rank0

PARALLEL_CONTEXTS = {}


def set_context(parallel_context: ParallelContext, name: str = "default"):
    PARALLEL_CONTEXTS[name] = parallel_context


def get_context(name: str = "default"):
    if name not in PARALLEL_CONTEXTS:
        raise KeyError(f"ParallelContext {name} is not initialized")

    return PARALLEL_CONTEXTS[name]
