"""Implementation of the Iris dataset.
"""

from .. import dataset


class Iris(dataset.Dataset):
  """Iris dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["target"])
