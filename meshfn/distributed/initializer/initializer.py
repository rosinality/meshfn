class ProcessGroupInitializer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        data_parallel_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
