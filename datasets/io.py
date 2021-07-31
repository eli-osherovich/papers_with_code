import abc
import pathlib
import tempfile
from typing import Union

import pandas as pd
import tensorflow as tf
import xopen

_DATASETS_DIR = 'pwc-datasets'


def download_dataset(
  ds_name: str,
  dl_config: dict[str, str],
  cache_dir: Union[str, pathlib.Path, None] = None,
):
  cache_dir = cache_dir or pathlib.Path(tempfile.gettempdir()) / _DATASETS_DIR
  # Make sure it's a Path.
  cache_dir = pathlib.Path(cache_dir)

  if ds_name not in dl_config:
    raise ValueError(
      f'Dataset name {ds_name} cannot be found in the download config {dl_config}'  # noqa E501
    )

  cache_dir.mkdir(parents=True, exist_ok=True)

  uri = dl_config[ds_name]['uri']
  checksum = dl_config[ds_name].get('checksum')

  data_path = tf.keras.utils.get_file(
    origin=uri, file_hash=checksum, cache_dir=cache_dir, cache_subdir='.'
  )
  return pathlib.Path(data_path)


class FileReader(abc.ABC):

  def __init__(self) -> None:
    super().__init__()

  @abc.abstractmethod
  def read(self, file_object):
    raise NotImplementedError


class FileAccessor():

  def __init__(self, path: Union[str, pathlib.Path], *args, **kwargs) -> None:
    self._path = path
    self._args = args
    self._kwargs = kwargs

  @property
  def path(self):
    return self._path

  def read(self, reader: FileReader):
    with xopen.xopen(self.path, *self._args, **self._kwargs) as f:
      return reader.read(f)


class PandasCSVReader(FileReader):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__()
    self._args = args
    self._kwargs = kwargs

  def read(self, file_object):
    return pd.read_csv(file_object, *self._args, **self._kwargs)