"""Implementation of the Iris dataset.
"""

from .. import dataset
from ..typing import DatasetType


class Iris(dataset.Dataset):
  """Iris dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.MULTICLASS, target_columns=["target"])
