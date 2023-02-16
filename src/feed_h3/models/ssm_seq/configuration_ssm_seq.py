from typing import (
    List,
    Optional
)
from dataclasses import (
    dataclass,
    field
)

from feed_h3.utils import Config


@dataclass
class SSMConfig(Config):
    head_dim: int = 1
    d_state: int = 64
    dropout: float = 0.0
    mode: str = 'diag'
    measure: str = 'diag-lin'
    use_fast_fftconv: bool = False


@dataclass
class AttnConfig(Config):
    num_heads: int = 12
    bias: bool = True
    dropout: float = 0.0
    rotary_emb_dim: Optional[int] = None

    def __post_init__(self):
        assert self.rotary_emb_dim in [None, 64], \
            'The `rotary_emb_dim` can either be `None`/`64`.'
        
        if self.rotary_emb_dim is None:
            self.rotary_emb_dim = 0


@dataclass
class SSMSeqConfig(Config):
    d_model: int = 768
    n_layer: int = 12
    ssm_cfg: SSMConfig = SSMConfig()
    attn_cfg: AttnConfig = AttnConfig()
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    d_inner: Optional[int] = None
    attn_layer_idx: Optional[List[int]] = field(
        default_factory=lambda: [6]
    )
    fused_mlp: bool = False
    fused_dropout_add_ln: bool = False

    def __post_init__(self):
        if self.d_inner is None:
            self.d_inner = 4 * self.d_model