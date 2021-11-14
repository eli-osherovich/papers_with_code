import importlib
import pathlib
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing as skl_preprocessing
import tensorflow as tf

from . import io
from . import utils
from .typing import DatasetType

DF_PAIR = tuple[pd.DataFrame, pd.DataFrame]
DS_PAIR = tuple[tf.data.Dataset, tf.data.Dataset]
NP_PAIR = tuple[np.ndarray, np.ndarray]

DS_RESULT = Union[DS_PAIR, tuple[DS_PAIR, ...]]
DF_RESULT = Union[DF_PAIR, tuple[DF_PAIR, ...]]
NP_RESULT = Union[NP_PAIR, tuple[NP_PAIR, ...]]


class DatasetFile:

  def __init__(
    self,
    *,
    uri: str,
    checksum: str = "",
    file_accessor_args: Optional[dict[str, Any]] = None,
    file_reader_args: Optional[dict[str, Any]] = None,
  ) -> None:
    self.uri = utils.make_uri(uri)
    self.checksum = checksum
    self.file_accessor_args = file_accessor_args or {}
    self.file_reader_args = file_reader_args or {}

  @property
  def cls_package(self) -> str:
    return self.__module__.rsplit(".", 1)[0]

  @property
  def cache(self) -> pathlib.Path:
    return utils.download_file(self.uri)

  def as_dataframe(self) -> pd.DataFrame:
    file_accessor = io.FileAccessor(self.cache, **self.file_accessor_args)
    file_reader = io.PandasCSVReader(**self.file_reader_args)
    return file_accessor.read(file_reader)


class Dataset:

  def __init__(
    self,
    *,
    type: DatasetType = DatasetType.RAW_DATA,
    target_columns=()
  ) -> None:
    self._type = type
    self._target_columns = target_columns
    self._cache = {}

  @property
  def cls_package(self) -> str:
    return self.__module__.rsplit(".", 1)[0]

  @property
  def ds_config(self):
    return importlib.import_module(".ds_config", self.cls_package)

  @property
  def splits(self):
    return self.ds_config.SPLITS

  def cache(self, split: str) -> list[pathlib.Path]:
    if split not in self.splits:
      raise RuntimeError(f"Requested split {split} does not exist")

    return self._cache.setdefault(split, self._download_files(split))

  # def as_dataset(self, *splits) -> DS_RESULT:
  #   res = tuple(self._generate_dataset(s) for s in splits)
  #   return squeeze(res)
  def as_dataframe(self, split: str) -> pd.DataFrame:
    data_frames = [
      f.as_dataframe() for f in utils.unsqueeze(self.splits[split])
    ]
    return pd.concat(data_frames, ignore_index=True)

  def as_preprocessed_dataframe(self, split: str) -> DF_PAIR:
    x = self.as_dataframe(split)
    y = x[self._target_columns]
    x = x.drop(self._target_columns, axis=1)
    x = pd.get_dummies(x, prefix_sep="__:__")
    if self._type in [DatasetType.BINARY, DatasetType.MULTICLASS
                     ] and len(self._target_columns) == 1:
      label_encoder = skl_preprocessing.LabelEncoder()
      y[self._target_columns[0]] = label_encoder.fit_transform(y.values.ravel())
    return x, y

  def as_preprocessed_numpy(self, split: str) -> NP_PAIR:
    x, y = self.as_preprocessed_dataframe(split)
    return x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)

  def _download_files(self, split: str):
    res = []
    for f in utils.unsqueeze(self.splits[split]):
      res.append(f.cache)
    return utils.squeeze(res)
