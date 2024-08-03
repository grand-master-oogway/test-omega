from __future__ import annotations

import logging
from enum import Enum
from .source import Source
from typing import Union, Dict, List
from dataclasses import dataclass
from .model_config import ModelConfig


class ConfigKeys(Enum):
    sources: Source = Source
    model_config: ModelConfig = ModelConfig

    def __getitem__(self, item) -> object:
        return getattr(self, item)


@dataclass
class Config:
    """
    Dataclass for sources and model configure.
    """

    data_json: Dict
    debug: bool = True

    def __init__(self, data_json: Dict, debug: bool = True):
        """
        Constructor for Config class.

        Args:
            data_json (Dict): Configuration data in JSON format.
            debug (bool): Flag to set the logging level to DEBUG if True, otherwise INFO.
        """
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.sources: Union[List[Source], None] = None
        self.model_config: Union[ModelConfig, None] = None
        self.data_json: Dict = data_json

    def __getitem__(self, item) -> object:
        """
        Get the attribute by key.

        Args:
            item (str): The key of the attribute to retrieve.

        Returns:
            object: The attribute value.
        """
        return getattr(self, item)

    @property
    def get_attributes(self) -> Union[Config, None]:
        """
        Get the attributes from the JSON configuration and set them to the instance.

        Returns:
            Union[Config, None]: The current instance with attributes set.
        """
        self._logger.debug('get attributes in Config')
        for key, value in self.data_json.items():
            if key == 'sources':
                self.sources = [Source(**value)]
            else:
                self.__setattr__(key, ConfigKeys.__getitem__(ConfigKeys, key).value(**value))
        return self
