"""Implementation of the Home Credit Default Risk dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class HomeCreditDefaultRisk(dataset.Dataset):
  """Home Credit Default Risk dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
