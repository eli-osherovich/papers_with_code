"""Implementation of the Iris dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class Iris(dataset.Dataset):
  """Iris dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.MULTICLASS, target_columns=["target"])
