import abc
import pathlib
import tempfile
from typing import Optional, Union
import zipfile

import pandas as pd
import tensorflow as tf
import xopen

_DATASETS_DIR = "pwc-datasets"


def download_dataset(
  ds_name: str,
  dl_config: dict[str, str],
  cache_dir: Union[str, pathlib.Path, None] = None,
) -> pathlib.Path:
  cache_dir = cache_dir or pathlib.Path(tempfile.gettempdir()) / _DATASETS_DIR
  # Make sure it's a Path.
  cache_dir = pathlib.Path(cache_dir)

  if ds_name not in dl_config:
    raise ValueError(
      f"Dataset name {ds_name} cannot be found in the download config {dl_config}"  # noqa E501
    )

  cache_dir.mkdir(parents=True, exist_ok=True)

  uri = dl_config[ds_name]["uri"]
  checksum = dl_config[ds_name].get("checksum")

  data_path = tf.keras.utils.get_file(
    origin=uri, file_hash=checksum, cache_dir=cache_dir, cache_subdir="."
  )
  return pathlib.Path(data_path)


class FileReader(abc.ABC):

  def __init__(self) -> None:
    super().__init__()

  @abc.abstractmethod
  def read(self, file_object):
    raise NotImplementedError


class FileAccessor:

  def __init__(
    self,
    path: Union[str, pathlib.Path],
    *,
    name: Union[None, str, pathlib.Path] = None,
    is_re: bool = False,
    **kwargs,
  ) -> None:
    self._path = pathlib.Path(path)
    self._kwargs = kwargs
    self._name = name
    self._is_re = is_re

  def xopen(self):
    if zipfile.is_zipfile(self._path):
      with zipfile.ZipFile(self._path) as zf:
        if self._is_re:
          name = [n for n in zf.namelist() if self._name in n][0]
        else:
          name = self._name
        return zf.open(name, **self._kwargs)
    else:
      return xopen.xopen(self._path, **self._kwargs)

  def read(self, reader: Optional[FileReader] = None):
    with self.xopen() as f:
      if reader is None:
        return f.read()
      else:
        return reader.read(f)


class PandasCSVReader(FileReader):

  def __init__(self, **kwargs) -> None:
    super().__init__()
    self._kwargs = kwargs

  def read(self, file_object) -> pd.DataFrame:
    return pd.read_csv(file_object, **self._kwargs)
