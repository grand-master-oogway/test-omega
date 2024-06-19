from .source import Source
from typing import Iterator, Any
from dataclasses import dataclass


@dataclass
class Config:
    def __init__(self, s_list):
        self.sources = [Source(i) for i in s_list]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.sources)
