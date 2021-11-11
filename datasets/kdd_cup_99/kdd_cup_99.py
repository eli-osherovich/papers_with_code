from .. import dataset


class KDDCup99(dataset.Dataset):
  """KDD Cup 99 dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["label"])
