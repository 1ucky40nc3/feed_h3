from feed_h3.utils import dataclass


def test_configclass():
    @dataclass
    class Cls():
        attr: str = 'val'
    
    cls = Cls()

    assert cls.attr == 'val'
    assert {**cls}['attr'] == 'val'
