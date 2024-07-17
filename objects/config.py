from __future__ import annotations

import logging
from enum import Enum
from typing import Union
from .source import Source
from dataclasses import dataclass
from .model_config import ModelConfig



class ConfigKeys(Enum):
    sources: Source = Source
    model_config: ModelConfig = ModelConfig
    def __getitem__(self, item) -> object:
        return getattr(self, item)


@dataclass
class Config:

    def __init__(self, data_json: dict, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.sources: Union[list[Source], None] = None
        self.model_config: Union[ModelConfig, None] = None
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
