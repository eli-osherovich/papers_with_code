"""Definition of the common ML tasks
"""
import enum

from absl import flags


@enum.unique
class TASK(enum.Enum):
  REGRESSION = 1
  BINARY = 2
  MULTICLASS = 3


flags.DEFINE_enum_class("task", TASK.REGRESSION, TASK, "Task type")
