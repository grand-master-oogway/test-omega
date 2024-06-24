from __future__ import annotations

import logging
from enum import Enum
from typing import Union
from .source import Source
from dataclasses import dataclass


class ConfigKeys(Enum):
    sources: Source = Source

    def __getitem__(self, item) -> object:
        return getattr(self, item)


@dataclass
class Config:

    def __init__(self, data_json: dict, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.sources: Union[Source, None] = None
        self.data_json: dict = data_json

    def __getitem__(self, item) -> object:
        return getattr(self, item)

    @property
    def get_attributes(self) -> Union['Config', None]:
        for key, value in self.data_json.items():
            self.__setattr__(key, ConfigKeys.__getitem__(ConfigKeys, key).value(**value))
        return self
