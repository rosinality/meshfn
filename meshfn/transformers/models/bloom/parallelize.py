from transformers.models.bloom.modeling_bloom import BloomModel

from meshfn.nn.parallel.strategy import (
    LinearColumn1D,
    LinearRow1D,
    VocabParallelEmbedding,
)
from meshfn.transformers.policy import Policy
from meshfn.transformers.models.bloom.modeling_bloom import (
    BloomParallelAttention,
    forward,
)


class BloomPolicy(Policy):
    layer_rules = {
        "*.self_attention": {"_module": BloomParallelAttention},
        "*.dense_h_to_4h": {"_strategy": LinearColumn1D},
        "*.dense_4h_to_h": {"_strategy": LinearRow1D},
        "*word_embeddings": {"_strategy": VocabParallelEmbedding},
        "*lm_head": {"_strategy": LinearColumn1D, "gather_output": True},
    }

    weight_rules = {
        "*.self_attention.query_key_value": {
            "_strategy": LinearColumn1D,
        },
        "*.self_attention.dense": {"_strategy": LinearRow1D},
    }

    attr_rules = {
        BloomModel: [{"_attr": "forward", "_target": forward}],
    }

    tie_rules = {"lm_head.weight": "transformer.word_embeddings.weight"}
