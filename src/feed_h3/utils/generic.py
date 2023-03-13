import dataclasses
from transformers import PretrainedConfig


def configclass(cls=None, **kwargs):
    def wrap(cls):
        def keys(self):
            return self.__dict__.keys()
        
        def __getitem__(self, key):
            return getattr(self, key)
        
        def pop(self, key, default=None):
            if key not in self.__dict__:
                return default
            return self.__dict__.pop()

        setattr(cls, 'pop', pop)
        setattr(cls, 'keys', keys)
        setattr(cls, '__getitem__', __getitem__)

        return  dataclasses.dataclass(cls)

    if cls is None:
        return wrap

    return wrap(cls)