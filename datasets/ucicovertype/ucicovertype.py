from .. import dataset


class UCICoverType(dataset.Dataset):
  """UCI Cover Type dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["CoverType"])
