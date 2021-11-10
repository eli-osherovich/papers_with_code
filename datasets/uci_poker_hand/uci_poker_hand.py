"""Implementation of the UCI Poker Hand dataset.
"""

from .. import dataset


class UCIPokerHand(dataset.Dataset):
  """UCI Poker Hand dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["Hand"], df_args={"skiprows": 0})
