"""Implementation of the UCI Adult dataset.
"""

from .. import dataset
from ..typing import DatasetType


class UCIAdult(dataset.Dataset):
  """UCI Adult dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
