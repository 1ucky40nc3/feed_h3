from typing import (
    List,
    Optional
)
from transformers import PretrainedConfig

from feed_h3.utils import configclass


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
        fused_dropout_add_ln = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        assert vocab_size is not None, 'The `vocab_size` needs to be provided!'

        # Set all but `self` and `kwargs` automatically
        args = locals()
        del args['self']
        del args['kwargs']
        for k, v in args.items():
            setattr(self, k, v)