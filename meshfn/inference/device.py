from typing import Dict, Optional, Union

import torch

DeviceType = Union[int, str, torch.device]


def build_device_maps(
    world_size: int, n_proc_per_node: int, rank: Optional[int] = None
) -> Dict[str, Dict[DeviceType, DeviceType]]:
    is_master = rank is None
    device_maps: Dict[str, Dict[DeviceType, DeviceType]] = {}

    if is_master:
        for i in range(world_size):
            worker_local_rank = i % n_proc_per_node
            device_maps[f"worker-{i}"] = {"cpu": worker_local_rank}

    else:
        local_rank = rank % n_proc_per_node

        for i in range(world_size):
            if i != rank:
                worker_local_rank = i % n_proc_per_node
                device_maps[f"worker-{i}"] = {local_rank: worker_local_rank}

        device_maps["master"] = {local_rank: "cpu"}

    return device_maps
