"""Implementation of the UCI Mushroom dataset.
"""

from .. import dataset


class UCIMushroom(dataset.Dataset):
  """UCI Mushroom dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["target"])
