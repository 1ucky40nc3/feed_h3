from feed_h3.utils import configclass


def test_configclass():
    @configclass
    class Cls():
        attr: str = 'val'
    
    cls = Cls()

    assert cls.attr == 'val'
    assert {**cls}['attr'] == 'val'
