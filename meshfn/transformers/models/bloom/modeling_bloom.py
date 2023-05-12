import math
import warnings
from typing import Tuple, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import BloomAttention
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from meshfn.distributed import ParallelContext, ParallelMode
from meshfn.nn.parallel.tensor1d import LinearColumn1D, LinearRow1D


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    powers = torch.arange(
        1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32
    )
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi


def _split_heads(
    fused_qkv: torch.Tensor, num_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    storage as `fused_qkv`

    Args:
        fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    fused_qkv = fused_qkv.view(batch_size, seq_length, num_heads, 3 * head_dim)
    query_layer, key_layer, value_layer = fused_qkv.split(head_dim, dim=-1)

    query_layer = query_layer.transpose(1, 2).reshape(
        batch_size * num_heads, seq_length, head_dim
    )
    key_layer = key_layer.permute(0, 2, 3, 1).reshape(
        batch_size * num_heads, head_dim, seq_length
    )
    value_layer = value_layer.transpose(1, 2).reshape(
        batch_size * num_heads, seq_length, head_dim
    )

    return query_layer, key_layer, value_layer


def _merge_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Merge heads together over the last dimenstion

    Args:
        x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

    Returns:
        torch.tensor: [batch_size, seq_length, num_heads * head_dim]
    """
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    # First view to decompose the batch size
    # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
    x = x.view(batch_size, num_heads, seq_length, head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = x.permute(0, 2, 1, 3)

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return x.reshape(batch_size, seq_length, num_heads * head_dim)


class BloomParallelAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        hidden_dropout,
        pretraining_tp,
        slow_but_exact,
        parallel_context,
    ):
        super().__init__()

        self.pretraining_tp = pretraining_tp
        self.slow_but_exact = slow_but_exact

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.num_heads = self.num_heads // parallel_context.world_size(
            ParallelMode.TENSOR_1D
        )

    @classmethod
    def from_module(cls, module: BloomAttention, parallel_context, **kwargs):
        attention = cls(
            module.hidden_size,
            module.num_heads,
            module.hidden_dropout,
            module.pretraining_tp,
            module.slow_but_exact,
            parallel_context,
        )
        attention.register_module(
            "query_key_value",
            LinearColumn1D.from_module(module.query_key_value, parallel_context),
        )
        attention.register_module(
            "dense", LinearRow1D.from_module(module.dense, parallel_context)
        )

        return attention

    @staticmethod
    def compute_attention(
        fused_qkv: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor],
        beta: float,
        inv_norm_factor: float,
        num_heads: int,
        use_cache: bool,
    ):
        batch_size, q_length, three_times_hidden_size = fused_qkv.shape
        head_dim = three_times_hidden_size // (3 * num_heads)
        batch_size_times_num_heads = batch_size * num_heads

        ### TODO @thomasw21: this takes quite a bit of time, how do I accelerate that?
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = _split_heads(
            fused_qkv, num_heads=num_heads, head_dim=head_dim
        )

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            past_key = past_key.view(-1, *past_key.shape[-2:])
            key_layer = torch.cat((past_key, key_layer), dim=2)
            past_value = past_value.view(-1, *past_value.shape[-2:])
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None
        ###

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=beta,
            alpha=inv_norm_factor,
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        # torch.finfo not supported by torch.jit, we temporarily remplace with `-1e34`
        attn_weights = attention_scores.masked_fill_(
            attention_mask, torch.finfo(attention_scores.dtype).min
        )
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            input_dtype
        )

        # # [batch_size, num_heads, q_length, kv_length]
        # attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = _merge_heads(
            context_layer, num_heads=num_heads, head_dim=head_dim
        )

        return context_layer, present, attention_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        batch_size, q_length, _ = fused_qkv.shape

        if layer_past is not None:
            past_key, past_value = layer_past
            layer_past = (
                past_key.view(-1, *past_key.shape[-2:]),
                past_value.view(-1, *past_value.shape[-2:]),
            )

        context_layer, present, attention_probs = self.compute_attention(
            fused_qkv=fused_qkv,
            layer_past=layer_past,
            alibi=alibi,
            attention_mask=attention_mask,
            head_mask=head_mask,
            beta=self.beta,
            inv_norm_factor=self.inv_norm_factor,
            num_heads=self.num_heads,
            use_cache=use_cache,
        )

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        # output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        output_tensor += residual

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **deprecated_arguments,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
    if deprecated_arguments.pop("position_ids", False) is not False:
        # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
        warnings.warn(
            "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
            " passing `position_ids`.",
            FutureWarning,
        )
    if len(deprecated_arguments) > 0:
        raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    hidden_states = self.word_embeddings_layernorm(inputs_embeds)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    # Compute alibi tensor: check build_alibi_tensor documentation
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[-1]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), device=hidden_states.device
        )
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    alibi = build_alibi_tensor(attention_mask, self.num_heads)

    causal_mask = self._prepare_attn_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        past_key_values_length=past_key_values_length,
    ).squeeze(1)

    if self.parallel_context.world_size(ParallelMode.TENSOR_1D) > 1:
        world_size = self.parallel_context.world_size(ParallelMode.TENSOR_1D)
        rank = self.parallel_context.local_rank(ParallelMode.TENSOR_1D)

        assert self.num_heads % world_size == 0
        block_size = self.num_heads // world_size
        alibi = alibi[:, rank * block_size : (rank + 1) * block_size]
        alibi = alibi.reshape(batch_size * block_size, 1, seq_length_with_past)
        causal_mask = torch.repeat_interleave(causal_mask, block_size, dim=0)

    else:
        alibi = alibi.reshape(batch_size * self.num_heads, 1, seq_length_with_past)
        causal_mask = torch.repeat_interleave(causal_mask, self.num_heads, dim=0)

    alibi = alibi.to(hidden_states.dtype)

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(
                        *inputs,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                alibi,
                causal_mask,
                head_mask[i],
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
