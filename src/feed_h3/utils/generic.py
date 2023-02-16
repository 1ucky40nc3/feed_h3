from dataclasses import dataclass


@dataclass
class Config:
    def keys(self):
        return self.__dict__.keys()

    def __setitem__(self, item, key):
        setattr(self, key, item)

    def __getitem__(self, key):
        return getattr(self, key)