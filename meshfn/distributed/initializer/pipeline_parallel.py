from torch import distributed as dist

from meshfn.distributed.parallel_mode import ParallelMode
from meshfn.distributed.initializer.initializer import ProcessGroupInitializer


class PipelineParallelInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_group_size = self.world_size // self.data_parallel_size
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_process_group(self, init_group=True):
        dist_settings = []

        pipe_group = None
        group_cpu = None

        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                pipe_ranks = list(
                    range(
                        i * self.data_group_size + j,
                        (i + 1) * self.data_group_size,
                        self.pipeline_stage_size,
                    )
                )

                if init_group:
                    pipe_group = dist.new_group(pipe_ranks)
                    group_cpu = (
                        dist.new_group(pipe_ranks, backend="gloo")
                        if dist.get_backend() != "gloo"
                        else pipe_group
                    )

                if self.rank in pipe_ranks:
                    local_rank = pipe_ranks.index(self.rank)
                    world_size = len(pipe_ranks)
                    process_group = pipe_group
                    cpu_group = group_cpu
                    ranks = pipe_ranks

                    dist_settings.append(
                        {
                            "local_rank": local_rank,
                            "world_size": world_size,
                            "process_group": process_group,
                            "cpu_group": cpu_group,
                            "ranks": ranks,
                            "mode": ParallelMode.PIPELINE,
                        }
                    )

        return dist_settings
