import enum
import sys

from absl import flags

from . import {{cookiecutter.model_prefix}}{{cookiecutter.model_module_suffix}}

FLAGS = flags.FLAGS

@enum.unique
class MODEL(enum.Enum):
  {{cookiecutter.model.upper()}} = 1

flags.DEFINE_enum_class("model", MODEL.{{cookiecutter.model.upper()}}, MODEL, "Model to use")


def get_model(**model_args):
  module_name = FLAGS.model.name.lower() + {{cookiecutter.model_module_suffix}}
  model_module = getattr(sys.modules[__package__], module_name)
  return model_module.get_model(**model_args)
