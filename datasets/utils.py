import inspect
import pathlib
from urllib import parse


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
