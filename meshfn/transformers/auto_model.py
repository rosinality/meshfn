import fnmatch
import importlib
import json
import os

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

from meshfn.distributed import ParallelMode
from meshfn.nn.parallel import parallelize
from meshfn.nn.load import init_empty_weights, load_checkpoint

POLICY_MAP = {"Bloom*": "bloom.parallelize.BloomPolicy"}


def get_policy(arch):
    for k, v in POLICY_MAP.items():
        if fnmatch.fnmatchcase(arch, k):
            module, policy = v.rsplit(".", 1)
            module = importlib.import_module(f"meshfn.transformers.models.{module}")

            return getattr(module, policy)

    raise ValueError(f"policy for {arch} is not found")


def materialize_meta(model):
    for n, p in model.named_parameters():
        if not p.is_meta:
            continue

        if "." in n:
            module, weight = n.rsplit(".", 1)
            module = model.get_submodule(module)

        else:
            module = model
            weight = p

        module.register_parameter(
            weight, nn.Parameter(torch.empty_like(p, device="cpu"))
        )


def auto_parallel_model(model_class, path, parallel_context):
    with init_empty_weights():
        model = model_class.from_config(AutoConfig.from_pretrained(path))

    policy = get_policy(model.__class__.__name__)

    model = parallelize(model, policy, parallel_context)
    materialize_meta(model)

    return model


def get_tp_pp(parallel_context):
    tp_size = parallel_context.tensor_parallel_size
    pp_size = parallel_context.pipeline_parallel_size

    tp_rank = parallel_context.local_rank(ParallelMode.TENSOR)

    try:
        pp_rank = parallel_context.local_rank(ParallelMode.PIPELINE)

    except KeyError:
        pp_rank = 0

    return tp_rank, tp_size, pp_rank, pp_size


def get_shard_name(tp_rank, pp_rank):
    return f"tp-{str(tp_rank).zfill(2)}_pp-{str(pp_rank).zfill(2)}"


def load_parallel_checkpoint(model, path, parallel_context):
    with open(os.path.join(path, "parallel_context.json")) as f:
        parallel_config = json.load(f)

    tp_rank, tp_size, pp_rank, pp_size = get_tp_pp(parallel_context)
    tp_size_conf = parallel_config["tensor_parallel_size"]
    pp_size_conf = parallel_config["pipeline_parallel_size"]

    if tp_size != tp_size_conf or pp_size != pp_size_conf:
        raise ValueError(
            (
                "parallelized configurations of checkpoint is differ from current parallel context."
                f" tensor parallel {tp_size} vs {tp_size_conf}, pipeline parallel {pp_size} vs {pp_size_conf}"
            )
        )

    shard_name = get_shard_name(tp_rank, pp_rank)
    path = os.path.join(path, shard_name)

    return load_checkpoint(model, path)


def auto_model_and_load_checkpoint(model_class, path, parallel_context):
    model = auto_parallel_model(model_class, path, parallel_context)

    return load_parallel_checkpoint(model, path, parallel_context)


class AutoParallelModel:
    @classmethod
    def from_pretrained(path, parallel_context):
        return auto_model_and_load_checkpoint(AutoModel, path, parallel_context)


class AutoParallelModelForCausalLM:
    @classmethod
    def from_pretrained(cls, path, parallel_context):
        return auto_model_and_load_checkpoint(
            AutoModelForCausalLM, path, parallel_context
        )
