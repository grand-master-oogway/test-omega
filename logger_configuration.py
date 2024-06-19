from __future__ import annotations

import os
import logging
from logging.handlers import TimedRotatingFileHandler


class LoggerConfigurator:
    """
    Contain common logger configuration

    Attributes:
        __default_logger            - reference to default logger (root logger)
        __default_message_format    - default log message format

    """

    def __init__(self):
        """
        Create new logger configuration
        """
        self.__default_logger = logging.getLogger()
        self.__default_logger.handlers = []
        self.__default_message_format = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"

    def set_default_format(self, message_format: str) -> LoggerConfigurator:
        """
        Set default message format

        Args:
            message_format: input format

        Returns:
            current instance
        """
        self.__default_message_format = message_format

        return self

    def set_level_debug(self) -> LoggerConfigurator:
        """
        Set logger minimum level - debug

        Returns:
            current instance
        """
        self.__default_logger.setLevel(logging.DEBUG)

        return self

    def set_level_info(self) -> LoggerConfigurator:
        """
        Set logger minimum level - info

        Returns:
            current instance
        """
        self.__default_logger.setLevel(logging.INFO)

        return self

    def set_level_warning(self) -> LoggerConfigurator:
        """
        Set logger minimum level - warning

        Returns:
            current instance
        """
        self.__default_logger.setLevel(logging.WARNING)

        return self

    def set_level(self, level) -> LoggerConfigurator:
        """
        Set custom logger minimum level

        Args:
            level: minimum log level

        Returns:
            current instance
        """
        self.__default_logger.setLevel(level)

        return self

    def add_console(self) -> LoggerConfigurator:
        """
        Add console handler to root logger

        Returns:
            current instance
        """

        formatter = logging.Formatter(self.__default_message_format)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.__default_logger.addHandler(handler)

        return self

    def add_timed_rotating(self, file_dir: str = "external/logs", rotation: str = "midnight", duration: int = 1,
                           max_copy: int = 7) -> LoggerConfigurator:
        """
        Add timed rotation file handler to root logger

        Args:
            file_dir: log file dir (optional).
                            Default - [external/logs].
            rotation: Log rotation type. (optional).
                            Default - [midnight]
            duration: log duration (optional).
                            Default - [1]
            max_copy: max log file copy count, to delete (optional).
                            Default - [7]
        Returns:
            current instance
        """
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        formatter = logging.Formatter(self.__default_message_format)
        handler = TimedRotatingFileHandler(f"{file_dir}/app.log",
                                           when=rotation, interval=duration, backupCount=max_copy)
        handler.setFormatter(formatter)
        handler.suffix = "%d-%m-%Y"
        self.__default_logger.addHandler(handler)

        return self
