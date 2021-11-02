import pandas as pd

from .. import dataset


class {{cookiecutter.dataset_class}}(dataset.Dataset):
  """{{cookiecutter.dataset}} dataset."""

  def __init__(self) -> None:
    super().__init__()
