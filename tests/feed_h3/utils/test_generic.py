from feed_h3.utils import dataclass


def test_configclass():
    @dataclass
    class A():
        b: int = 1
    
    a = A()

    assert a.b == 1
    assert {**a}['b'] == 1
