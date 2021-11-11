from collections.abc import Sequence
import inspect
import pathlib
import tempfile
from typing import Optional, Union
from urllib import parse

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
) -> pathlib.Path:
  cache_dir = cache_dir or pathlib.Path(tempfile.gettempdir())
  cache_dir = pathlib.Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)

  return tf.keras.utils.get_file(
    origin=uri, file_hash=checksum, cache_dir=cache_dir, cache_subdir="."
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
