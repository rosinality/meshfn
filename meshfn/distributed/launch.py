import os
from typing import Optional, Union

from meshfn.distributed import ParallelMode, ParallelContext, set_context
from meshfn.logging import get_logger


def launch(
    rank: int,
    world_size: int,
    host: str,
    port: int,
    local_rank: Optional[int] = None,
    backend: str = "nccl",
    tensor_parallel_size: int = 1,
    tensor_parallel_mode: Optional[Union[str, ParallelMode]] = None,
    pipeline_parallel_size: int = 1,
    seed: int = 1024,
    verbose: bool = True,
):
    context = ParallelContext(
        rank,
        world_size,
        host,
        port,
        local_rank,
        backend,
        tensor_parallel_size,
        tensor_parallel_mode,
        pipeline_parallel_size,
        seed,
        verbose,
    )

    if verbose:
        logger = get_logger(parallel_context=context)
        logger.info(
            "INIT Initialized distributed environments; "
            f"data parallel size: {context.data_parallel_size},"
            f" tensor parallel size: {context.tensor_parallel_size} mode: {context.tensor_parallel_mode},"
            f" pipeline parallel size: {context.pipeline_parallel_size}",
            ranks=[0],
        )

    return context


def from_torch(
    backend: str = "nccl",
    tensor_parallel_size: int = 1,
    tensor_parallel_mode: Optional[Union[str, ParallelMode]] = None,
    pipeline_parallel_size: int = 1,
    seed: int = 1024,
    verbose: bool = True,
):
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])

    except KeyError as e:
        raise RuntimeError(
            f"could not find {e} in environment variables set by torch launcher"
        )

    return launch(
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        local_rank=local_rank,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_mode=tensor_parallel_mode,
        pipeline_parallel_size=pipeline_parallel_size,
        seed=seed,
        verbose=verbose,
    )
