""" Config variables.

If you need to change configuration, copy this file into your group directory
and import it from there.
"""
from dataclasses import dataclass
import logging


@dataclass
class Config:
    LOG_LEVEL: int = (  # Change to logging.ERROR to disable debug.
        logging.DEBUG
    )
    PLAYBACK: bool = True
    LOG_DIRECTORY: str = "./logs/"


config = Config()
