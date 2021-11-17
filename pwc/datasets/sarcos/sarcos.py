"""Implementation of the Sarcos dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class Sarcos(dataset.Dataset):
  """Sarcos dataset."""

  def __init__(self) -> None:
    super().__init__(
      type=DatasetType.REGRESSION,
      target_columns=["r22", "r23", "r24", "r25", "r26", "r27", "r28"]
    )
