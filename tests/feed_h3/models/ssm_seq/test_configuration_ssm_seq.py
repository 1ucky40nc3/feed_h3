from feed_h3 import (
    SSMConfig,
    AttnConfig,
    SSMSeqConfig
)


TEST_PATH = 'tests/feed_h3/models/ssm_seq'


def test_ssm_config():
    config = SSMConfig()


def test_attn_config():
    config = AttnConfig()


def test_ssm_seq_config():
    vocab_size = 1
    config = SSMSeqConfig(vocab_size=vocab_size)
    config.save_pretrained(TEST_PATH)
    config = SSMSeqConfig.from_pretrained(TEST_PATH)
    assert str(config) == str(SSMSeqConfig())

    config = SSMSeqConfig(n_embd=10)
    assert config.n_embd == 10