from typing import (
    List,
    Optional
)
from transformers import PretrainedConfig

from ...utils.generic import configclass


UNDEFINED = '__UNDEFINED__'


@configclass
class SSMConfig:
    head_dim: int = 1
    d_state: int = 64
    dropout: float = 0.0
    mode: str = 'diag'
    measure: str = 'diag-lin'
    use_fast_fftconv: bool = False


@configclass
class AttnConfig:
    num_heads: int = 12
    bias: bool = True
    dropout: float = 0.0
    rotary_emb_dim: Optional[int] = None

    def __post_init__(self):
        assert self.rotary_emb_dim in [None, 0, 64], \
            'The `rotary_emb_dim` can either be `None`/`64`.'
        
        if self.rotary_emb_dim is None:
            self.rotary_emb_dim = 0


class SSMSeqConfig(PretrainedConfig):
    def __init__(
        self,
        n_embd: int = 768,
        n_inner: Optional[int] = None,
        n_layer: int = 12,
        attn_layer_idx: List[int] = [6],
        attn_n_head: Optional[int] = 12,
        attn_pdrop: float = 0.0,
        attn_rotary_dim: Optional[int] = None,
        attn_bias: bool = True,
        ssm_head_dim: int = 1,
        ssm_d_state: int = 64,
        ssm_pdrop: float = 0.0,
        ssm_mode: str = 'diag',
        ssm_measure: str = 'diag-lin',
        ssm_use_fast_fttconv: bool = False,
        resid_pdrop: float = 0.0,
        embed_pdrop: float = 0.1,
        vocab_size: int = None,
        layer_norm_epsilon: float = 1e-5,
        fused_mlp: bool = False,
        fused_dropout_add_ln: bool = False,
        initializer_range: float = 0.02,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        **kwargs
    ) -> None:
        self.n_embd = n_embd
        self.n_inner = n_inner
        self.n_layer = n_layer
        self.attn_layer_idx = attn_layer_idx
        self.attn_n_head = attn_n_head
        self.attn_pdrop = attn_pdrop
        self.attn_rotary_dim = attn_rotary_dim
        self.attn_bias = attn_bias
        self.ssm_head_dim = ssm_head_dim
        self.ssm_d_state = ssm_d_state
        self.ssm_pdrop = ssm_pdrop
        self.ssm_mode = ssm_mode
        self.ssm_measure = ssm_measure
        self.ssm_use_fast_fttconv = ssm_use_fast_fttconv
        self.resid_pdrop = resid_pdrop
        self.embed_pdrop = embed_pdrop
        self.vocab_size = vocab_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.fused_mlp = fused_mlp
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.initializer_range = initializer_range

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # Set all but `self` and `kwargs` automatically
        # args = locals()
        # del args['self']
        # del args['kwargs']
        # for k, v in args.items():
        #     setattr(self, k, v)
        
        # assert vocab_size is not None, 'The `vocab_size` needs to be provided!'