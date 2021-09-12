from .. import dataset


class MillimanDataset(dataset.Dataset):
  """Milliman dataset."""

  def __init__(self) -> None:
    super().__init__(
      target_columns=["default"],
      df_args={
        "sep": "\\s+,",
        "header": 0,
        "engine": "python",
        "true_values": ["yes"],
        "false_values": ["no"],
      }
    )
