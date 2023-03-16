from torch import distributed as dist

from meshfn.distributed.parallel_mode import ParallelMode
from meshfn.distributed.initializer.tensor_parallel import TensorParallelInitializer


class TensorParallel1DInitializer(TensorParallelInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_tensor_parallel_1d_group = self.world_size // self.tensor_parallel_size

    def init_process_group(self):
        group = super().init_process_group()
        group["mode"] = ParallelMode.TENSOR_1D

        return group
