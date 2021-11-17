from absl import logging

from . import *  # noqa F403
from . import dataset


# TODO: consider class registry instead of blind import.
def cls_factory(cls, **kwargs):
  try:
    obj = globals()[cls](**kwargs)
    return obj
  except KeyError as e:
    logging.error("Cannot find DS class %s", cls)
    raise ValueError(f"Unable to load class called {cls}") from e


def load_dataframe(ds_cls, split: str, **kwargs):
  ds = cls_factory(ds_cls, **kwargs)
  return ds.as_dataframe(split)


def load_preprocessed_dataframe(ds_cls, split: str, **kwargs):
  ds = cls_factory(ds_cls, **kwargs)
  return ds.as_preprocessed_dataframe(split)


def load_preprocessed_numpy(ds_cls, split: str, **kwargs):
  ds = cls_factory(ds_cls, **kwargs)
  return ds.as_preprocessed_numpy(split)
