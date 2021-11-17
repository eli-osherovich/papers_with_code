import pathlib

import gin

from .model import get_model  # noqa F401
from .model import MODEL

_GIN_CONFIG_FILE = "config.gin"

gin.parse_config_file(
  pathlib.Path(__file__).parent.resolve() / _GIN_CONFIG_FILE
)
