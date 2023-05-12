from torch import distributed as dist

from meshfn.distributed.parallel_mode import ParallelMode
from meshfn.distributed.initializer.initializer import ProcessGroupInitializer


class TensorParallelInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_tensor_parallel_1d_group = self.world_size // self.tensor_parallel_size

    def init_process_group(self, init_group=True):
        local_rank = None
        ranks = None
        process_group = None
        cpu_group = None
        world_size = None

        group = None
        group_cpu = None

        for i in range(self.n_tensor_parallel_1d_group):
            ranks_list = [
                i * self.tensor_parallel_size + j
                for j in range(self.tensor_parallel_size)
            ]

            if init_group:
                group = dist.new_group(ranks_list)
                group_cpu = (
                    dist.new_group(ranks_list, backend="gloo")
                    if dist.get_backend() != "gloo"
                    else group
                )

            if self.rank in ranks_list:
                local_rank = ranks_list.index(self.rank)
                world_size = len(ranks_list)
                process_group = group
                cpu_group = group_cpu
                ranks = ranks_list

        return {
            "local_rank": local_rank,
            "world_size": world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks": ranks,
            "mode": ParallelMode.TENSOR,
        }
