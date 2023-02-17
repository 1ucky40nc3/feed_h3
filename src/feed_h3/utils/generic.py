import dataclasses


def dataclass(cls=None, **kwargs):
    def wrap(cls):
        def keys(self):
            return self.__dict__.keys()
        
        def __getitem__(self, key):
            return getattr(self, key)
        
        setattr(cls, 'keys', keys)
        setattr(cls, '__getitem__', __getitem__)

        return dataclasses.dataclass(cls)

    if cls is None:
        return wrap

    return wrap(cls)