import abc
import pathlib
from typing import Optional, Union
import zipfile

import pandas as pd
import xopen


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
    self._name = str(name)
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
