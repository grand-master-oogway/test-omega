from __future__ import annotations

import logging
from enum import Enum
from .source import Source
from typing import Union, List
from dataclasses import dataclass, fields


class ConfigKeys(Enum):
    sources: Source = Source

    def __getitem__(self, item) -> object:
        return getattr(self, item)


@dataclass
class Config:

    def __init__(self, data_json: dict, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.sources: Union[list[Source], None] = None
        self.data_json: dict = data_json

    def __getitem__(self, item) -> object:
        return getattr(self, item)

    @property
    def get_attributes(self) -> Union['Config', None]:
        self._logger.debug('get attributes in Config')
        for key, value in self.data_json.items():
            if key == 'sources':
                self.sources = [Source(**value)]
            else:
                self.__setattr__(key, ConfigKeys.__getitem__(ConfigKeys, key).value(**value))
        return self
