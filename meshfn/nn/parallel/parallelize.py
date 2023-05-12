import fnmatch
import itertools
from types import MethodType
from typing import Dict, Any, List

from torch import nn

from meshfn.distributed import ParallelContext


def distribute_tensor(tensor, placement, parallel_context):
    shards = []
    parallel_modes = []

    for place in placement:
        if place is None:
            shard = 1

        else:
            shard = place.n_shard(parallel_context)
            parallel_modes.append(place.parallel_mode)

        shards.append(shard)

    places_list = []
    for shard in shards:
        if shard < 2:
            continue

        places_list.append(list(range(shard)))

    places = list(itertools.product(*places_list))

    tensor_shards = [tensor]
    for dim, shard in enumerate(shards):
        t_shard = []
        for target in tensor_shards:
            if shard != 1:
                target = placement[dim].place(target, dim, parallel_context)

            else:
                target = [target]

            t_shard.extend(target)

        tensor_shards = t_shard

    return dict(zip(places, tensor_shards)), parallel_modes


def get_shard(shards, parallel_modes, parallel_context: ParallelContext):
    shard_key = tuple([parallel_context.local_rank(mode) for mode in parallel_modes])

    return shards[shard_key]


def shard_layer_state_dict(
    state_dict, key, layer_strategy, parallel_context: ParallelContext
):
    sharded = {}
    non_sharded = {}

    for name, strategy in layer_strategy.strategy.items():
        path = f"{key}.{name}"

        try:
            tensor = state_dict[path]

        except KeyError:
            if strategy.optional:
                continue

            raise KeyError(f"non-optional key {path} is not found on state_dict")

        shards, modes = distribute_tensor(tensor, strategy.placement, parallel_context)

        if len(modes) > 0:
            sharded[path] = shards

        else:
            non_sharded[path] = shards[()]

    return sharded, non_sharded


def shard_state_dict(state_dict, mapping, parallel_context: ParallelContext):
    new_state_dict = {}
    sharded_state_dict = {}

    layer_rules = {
        k: v for k, v in mapping.get_layer_rules().items() if "_strategy" in v
    }
    mapping = {**layer_rules, **mapping.get_weight_rules()}

    mapping_pattern = {
        k: v for k, v in mapping.items() if isinstance(k, str) and "_strategy" in v
    }

    for key, tensor in state_dict.items():
        sharded = False

        for pattern, convert in mapping_pattern.items():
            strategy = convert["_strategy"]
            targets = strategy.strategy.keys()

            for target in targets:
                if fnmatch.fnmatchcase(key, f"{pattern}.{target}"):
                    layer_key = key.rsplit(".", 1)[0]

                    sharded_dict, non_sharded_dict = shard_layer_state_dict(
                        state_dict, layer_key, strategy, parallel_context
                    )
                    sharded_state_dict.update(sharded_dict)
                    new_state_dict.update(non_sharded_dict)

                    sharded = True

        if not sharded:
            new_state_dict[key] = tensor

    return new_state_dict, sharded_state_dict


def combine_shard_state_dict(nonshard_state_dict, shard_state_dict, sort_key=None):
    state_dicts = {}

    for k, shards in shard_state_dict.items():
        for shard_i, tensor in shards.items():
            # Here I assumes shard indexes corresponds to tensor parallel (not tensor 1d) ranks.
            # I will change rest of sharding mechanism to support multiple sharding.
            shard_i = shard_i[0]

            if shard_i not in state_dicts:
                state_dicts[shard_i] = {}

            state_dicts[shard_i][k] = tensor.clone()

    for shard_i in state_dicts:
        for k, tensor in nonshard_state_dict.items():
            state_dicts[shard_i][k] = tensor.clone()

    if sort_key is not None:
        sorted_dict = {}

        for shard_i, state in state_dicts.items():
            sorted_dict[shard_i] = {}

            for key in sort_key:
                sorted_dict[shard_i][key] = state[key]

        state_dicts = sorted_dict

    return state_dicts


def parallelize(
    module: nn.Module,
    mapping: Dict[Any, Dict[str, Any]],
    parallel_context: ParallelContext,
):
    module_names: List[str] = []

    # update attributes/methods
    for m in module.modules():
        for target, converts in mapping.get_attr_rules().items():
            if not isinstance(m, target):
                continue

            m.parallel_context = parallel_context

            for convert in converts:
                _attr = convert["_attr"]
                _target = convert["_target"]
                attr = getattr(m, _attr)

                if isinstance(attr, MethodType):
                    setattr(m, _attr, MethodType(_target, m))

                else:
                    setattr(m, _attr, _target)

    for n, _ in module.named_modules():
        module_names.append(n)

    module_patch: Dict[str, Dict[str, Any]] = {}

    for name in module_names:
        for pattern, convert in mapping.get_layer_rules().items():
            if fnmatch.fnmatchcase(name, pattern):
                module_patch[name] = convert

    # replace layers
    for path, patch in module_patch.items():
        if "." in path:
            parent, child = path.rsplit(".", 1)
            parent_module = module.get_submodule(parent)
            child_module = module.get_submodule(path)

        else:
            parent_module = module
            child_module = module.get_submodule(path)
            child = path

        patch = patch.copy()

        if "_module" in patch:
            convert = patch.pop("_module")

        elif "_strategy" in patch:
            convert = patch.pop("_strategy").layer

        convert = convert.from_module(
            child_module, parallel_context=parallel_context, **patch
        )
        parent_module.register_module(child, convert)

    # tie weights
    for tie_target, tie_orig in mapping.get_tie_rules().items():
        target_module, target_name = tie_target.rsplit(".", 1)
        orig_module, orig_name = tie_orig.rsplit(".", 1)

        target_module = module.get_submodule(target_module)
        orig_module = module.get_submodule(orig_module)

        setattr(target_module, target_name, getattr(orig_module, orig_name))

    return module
