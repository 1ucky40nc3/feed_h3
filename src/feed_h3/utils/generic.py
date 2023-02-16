from typing import Any
from dataclasses import dataclass


dict_keys = Any


@dataclass
class Config:
    def keys(self) -> dict_keys[str, Any]:
        return self.__dict__.keys()

    def __setitem__(self, item: Any, key: str) -> None:
        setattr(self, key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)