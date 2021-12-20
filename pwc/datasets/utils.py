from collections.abc import Sequence
import inspect
import pathlib
import tempfile
from typing import Optional, Union
from urllib import parse

import numpy as np
import pandas as pd
import tensorflow as tf


def make_uri(path):
  # If the path has a schema already, we assume that it is a URI.
  if parse.urlparse(path).scheme:
    return path

  # Otherwise, we assume this is a local file
  # Absolute paths can be converted to an URI,
  # For relative paths we assume that they are relative to the caller's file.
  path = pathlib.Path(path)
  if path.is_absolute():
    return path.as_uri()
  else:
    caller_path = pathlib.Path(inspect.stack()[1].filename)
    return (caller_path.parent / path).as_uri()


def download_file(
  uri: Union[str, pathlib.Path],
  checksum: Optional[str] = None,
  *,
  cache_dir: Union[str, pathlib.Path, None] = None,
  cache_subdir: Union[str, pathlib.Path] = ".",
) -> pathlib.Path:
  cache_dir = cache_dir or pathlib.Path(tempfile.gettempdir())
  cache_dir = pathlib.Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)

  return tf.keras.utils.get_file(
    origin=uri,
    file_hash=checksum,
    cache_dir=cache_dir,
    cache_subdir=pathlib.Path(cache_subdir),
  )


def squeeze(seq: Sequence):
  if len(seq) == 1:
    return seq[0]
  else:
    return seq


def unsqueeze(obj) -> Sequence:
  if isinstance(obj, Sequence):
    return obj
  else:
    return [obj]


def pandas_downcast(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
  """Convert numerical types to the 'minimal' sufficient type"""

  if not inplace:
    df = df.copy()

  float_cols = df.select_dtypes("float").columns
  int_cols = df.select_dtypes("integer").columns

  df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
  df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")

  is_boolean = df.select_dtypes(["int8", "uint8"]).isin([0, 1]).all()
  bool_cols = is_boolean[is_boolean].index
  df[bool_cols] = df[bool_cols].astype(bool)

  # Some float columns are boolean with missing values.
  # Pandas supports its own nullable boolean type.
  is_boolean = df[float_cols].isin([0, 1, np.nan]).all()
  bool_cols = is_boolean[is_boolean].index
  df[bool_cols] = df[bool_cols].astype("boolean")

  return df
