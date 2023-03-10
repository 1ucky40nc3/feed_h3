from typing import (
    Union,
    Tuple,
    Optional
)
import re
from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.pytorch_utils import Conv1D

from h3.models.ssm_seq import SSMModel

from flash_attn.utils.generation import (
    GenerationMixin,
    InferenceParams
)

from .configuration_ssm_seq import (
    SSMSeqConfig, 
    SSMConfig, 
    AttnConfig
)

@dataclass
class CausalLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None


class SSMPretrainedModel(PreTrainedModel):
    config_class = SSMSeqConfig
    base_model_prefix = "feed_h3"
    is_parallelizable = False
    supports_gradient_checkpointing = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SSMLMHeadModel(SSMPretrainedModel, GenerationMixin):
    def __init__(self, config, pad_vocab_size_to_multiple_of: int = 1):
        super().__init__(config)

        # if config.vocab_size % pad_vocab_size_to_multiple_of != 0:
        #     self.config.vocab_size += pad_vocab_size_to_multiple_of \
        #         - (config.vocab_size % pad_vocab_size_to_multiple_of)

        print(f'{config=}')

        self.ssm_cfg = SSMConfig(
            head_dim=config.ssm_head_dim,
            d_state=config.ssm_d_state,
            dropout=config.ssm_pdrop,
            mode=config.ssm_mode,
            measure=config.ssm_measure,
            use_fast_fftconv=config.ssm_use_fast_fttconv
        )
        self.attn_cfg = AttnConfig(
            num_heads=config.attn_n_head,
            bias=config.attn_bias,
            dropout=config.attn_pdrop,
            rotary_emb_dim=config.attn_rotary_dim
        )

        self.backbone = SSMModel(
            d_model=config.n_embd,
            n_layer=config.n_layer,
            d_inner=config.n_inner,
            vocab_size=config.vocab_size,
            ssm_cfg=self.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=self.attn_cfg,
            resid_dropout=config.resid_pdrop,
            embed_dropout=config.embed_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon,
            fused_mlp=config.fused_mlp,
            fused_dropout_add_ln=config.fused_dropout_add_ln
        )

        self.lm_head = nn.Linear(
            in_features=self.config.n_embd,
            out_features=self.config.vocab_size,
            bias=False
        )

        self.post_init()

    def _tie_weights(self):
        self.lm_weight = self.backbone.embeddings.word_embeddings.weight
    
    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used different names
        def key_mapping_backbone(key):
            key = re.sub(r'^s4seq.encoder.', 'backbone.', key)
            key = re.sub(r'^embedding.', 'backbone.embeddings.word_embeddings.', key)
            key = re.sub(r'^backbone.norm', 'backbone.ln_0', key)
            return key
        state_dict = OrderedDict((key_mapping_backbone(k), v) for k, v in state_dict.items())
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Mixer / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if 'backbone.ln_0.weight' in state_dict:
            n_layers = len(self.backbone.layers)
            ln_weight = state_dict.pop(f'backbone.layers.{n_layers - 1}.norm2.weight')
            ln_bias = state_dict.pop(f'backbone.layers.{n_layers - 1}.norm2.bias')
            state_dict['backbone.ln_f.weight'] = ln_weight
            state_dict['backbone.ln_f.bias'] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f'backbone.layers.{l}.norm1.weight')
                ln_bias = state_dict.pop(f'backbone.layers.{l}.norm1.bias')
                state_dict[f'backbone.layers.{l}.norm2.weight'] = ln_weight
                state_dict[f'backbone.layers.{l}.norm2.bias'] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f'backbone.layers.{l - 1}.norm2.weight')
                    ln_bias = state_dict.pop(f'backbone.layers.{l - 1}.norm2.bias')
                    state_dict[f'backbone.layers.{l}.norm1.weight'] = ln_weight
                    state_dict[f'backbone.layers.{l}.norm1.bias'] = ln_bias
            ln_weight = state_dict.pop('backbone.ln_0.weight')
            ln_bias = state_dict.pop('backbone.ln_0.bias')
            state_dict[f'backbone.layers.0.norm1.weight'] = ln_weight
            state_dict[f'backbone.layers.0.norm1.bias'] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params: Optional[InferenceParams] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        hidden_states = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            inference_params=inference_params
        )

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Logits with shape: (batch_size, block_size, vocab_size)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            # Labels with shape: (batch_size,)
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (lm_logits, hidden_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states
        )