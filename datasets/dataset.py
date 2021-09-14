import importlib

import numpy as np
import pandas as pd
import tensorflow as tf

from . import io


class Dataset:

  def __init__(self, *, target_columns=(), df_args=None) -> None:
    self._target_columns = target_columns
    self._df_args = df_args or {}

  @property
  def cls_package(self):
    return self.__module__.rsplit(".", 1)[0]

  @property
  def config_datasets(self):
    dl_config = importlib.import_module(".dl_config", self.cls_package)
    return dl_config.DATASETS

  @property
  def feature_dict(self):
    ds_config = importlib.import_module(".ds_config", self.cls_package)
    return ds_config.feature_dict

  def download_dataset(self, ds_name):
    return io.download_dataset(ds_name, self.config_datasets)

  def as_dataset(self, *splits):
    res = tuple(self._generate_dataset(s) for s in splits)
    return squeeze(res)

  def as_dataframe(self, *splits):
    res = tuple(self._generate_dataframe(s) for s in splits)
    return squeeze(res)

  def as_numpy(self, *splits):
    res = tuple(self._generate_numpy(s) for s in splits)
    return squeeze(res)

  def _generate_dataframe(self, split_name):
    X, y = self._read_df(split_name)
    X = pd.get_dummies(X, prefix_sep="__:__")
    return X, y

  def _read_df(self, split_name):
    ds_path = self.download_dataset(split_name)
    file_accessor = io.FileAccessor(ds_path)
    file_reader = io.PandasCSVReader(
      names=self.feature_dict, dtype=self.feature_dict, **self._df_args
    )
    X = file_accessor.read(file_reader)
    y = X[self._target_columns]
    X = X.drop(self._target_columns, axis=1)
    return X, y

  def _generate_numpy(self, split_name):
    dataframes = self._generate_dataframe(split_name)
    if isinstance(dataframes, (tuple, list)):
      arrays = tuple(df.to_numpy(dtype=np.float32) for df in dataframes)
    elif isinstance(dataframes, pd.DataFrame):
      arrays = dataframes.to_numpy(dtype=np.float32)
    return arrays

  def _generate_dataset(self, split_name):
    arrays = self._generate_numpy(split_name)
    return tf.data.Dataset.from_tensor_slices(arrays)


def squeeze(vec):
  if len(vec) == 1:
    return vec[0]
  else:
    return vec
