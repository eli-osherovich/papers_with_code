import pathlib

import gin

from .training import training_fn  # noqa F401

_GIN_CONFIG_FILE = "config.gin"

gin.parse_config_file(
  pathlib.Path(__file__).parent.resolve() / _GIN_CONFIG_FILE
)