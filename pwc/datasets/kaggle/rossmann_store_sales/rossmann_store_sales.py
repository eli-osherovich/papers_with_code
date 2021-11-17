"""Implementation of the Rossmann Store Sales dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class RossmannStoreSales(dataset.Dataset):
  """Rossmann Store Sales dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.REGRESSION, target_columns=["Sales"])
