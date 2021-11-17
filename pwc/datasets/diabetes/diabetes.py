"""Implementation of the Diabetes dataset.
"""

from .. import dataset
from ..typing import DatasetType


class Diabetes(dataset.Dataset):
  """Diabetes dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.REGRESSION, target_columns=["target"])
