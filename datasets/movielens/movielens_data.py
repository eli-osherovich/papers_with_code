import functools
import logging
import os
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf

from . import movielens_config
from ..common import utils


class MovieLens:

  def __init__(self, zfile: str) -> None:
    assert os.access(
      zfile, os.R_OK), f"File {zfile} either does not exist or not readable."
    assert zipfile.is_zipfile(zfile), f"File {zfile} is not a valid ZIP file."
    self.zfile = zfile
    self.items_files = ["movies.csv", "movies.dat"]
    self.ratings_files = ["ratings.csv", "ratings.dat"]

  @staticmethod
  def sniff_csv_config(fd):
    pos = fd.tell()
    fd.seek(0)
    line = fd.readline()
    fd.seek(pos)
    if b"::" in line:
      sep = "::"
      header = None
      engine = "python"
    else:
      sep = ","
      header = 0
      engine = "c"
    return sep, header, engine

  def read_csv(self, file_names, **kwargs):
    with zipfile.ZipFile(self.zfile) as zfile:
      for file_info in zfile.filelist:
        if os.path.basename(file_info.filename) in file_names:
          break
      else:
        raise RuntimeError(f"Cannot find files {file_names} in {self.zfile}")
      with zfile.open(file_info) as fd:
        sep, header, engine = self.sniff_csv_config(fd)
        df = pd.read_csv(
          fd,
          sep=sep,
          header=header,
          engine=engine,
          encoding="latin1",
          **kwargs)

        # Keep only clean data.
        df.dropna(inplace=True)
        return df

  @functools.cached_property
  def ratings(self) -> pd.DataFrame:
    # We do not use timestamp, hence we do not read it.
    # Column types are explicitly set to lower memory consumption.
    # Furthermore, we use `user` as index to save an extra column.
    df = self.read_csv(
      self.ratings_files,
      names=("user", "item", "rating", "timestamp"),
      usecols=("user", "item", "rating"),
      index_col="user",
      dtype={
        "user": np.uint32,  # BUG: pandas ignores index type
        "item": np.uint32,
        "rating": np.float32
      })

    # Keep only clean data.
    df.dropna(inplace=True)
    return df

  @functools.cached_property
  def items(self) -> pd.DataFrame:
    # Actual columns are: `item`, `title`, `genres`.
    # However, we do not use `title`, we only extract release date from it.
    # Hence, we do not read it into the table.
    df = self.read_csv(
      self.items_files,
      names=("item", "release year", "genres"),
      dtype={
        "item": np.uint32,
        "genres": str,
      },
      index_col="item",
      converters={
        "release year":
          lambda x: pd.to_numeric(x.rstrip()[-5:-1], errors="coerce")
      })

    # Convert genres to dummy variables.
    genres = df.pop("genres")
    genres = pd.get_dummies(
      genres.map(lambda x: x.split("|")).explode()).sum(level=0)
    df = pd.concat([df, genres], axis=1)

    # Keep only clean data.
    df.dropna(inplace=True)
    return df

  def expand_items_data(self):
    items_data = self.ratings.groupby("item").rating.agg([
      "mean",
      "count",
    ]).rename(columns={
      "mean": "item_ratings_mean",
      "count": "item_ratings_count",
    }).astype(np.float32)

    items_data = items_data.merge(
      self.items, how="inner", left_index=True, right_index=True, copy=False)
    del self.items
    return items_data

  def expand_users_data(self):
    users_data = self.ratings.groupby("user").rating.agg([
      "mean",
      "count",
    ]).rename(columns={
      "mean": "user_ratings_mean",
      "count": "user_ratings_count",
    }).astype(np.float32)

    users_data = self.ratings.merge(
      users_data, how="inner", left_index=True, right_index=True, copy=False)
    del self.ratings
    return users_data

  @functools.cached_property
  def full_table(self) -> pd.DataFrame:

    items_data = self.expand_items_data()
    users_data = self.expand_users_data()
    df = users_data.merge(
      items_data, how="inner", left_on="item", right_index=True, copy=False)

    # Avoid data leakage: remove current rating from aggregated user/movie data
    # In the following we use the formula:
    # (x*n - y)/(n-1) = x + (x-y)/(n-1)
    df["user_ratings_count"] -= 1
    df["item_ratings_count"] -= 1

    df["user_ratings_mean"] += ((df["user_ratings_mean"] - df["rating"]) /
                                df["user_ratings_count"])

    df["item_ratings_mean"] += ((df["item_ratings_mean"] - df["rating"]) /
                                df["item_ratings_count"])

    # Keep only clean data.
    df.dropna(inplace=True)
    return df


def download_dataset(ds_name="latest-small", cache_dir="datasets"):
  os.makedirs(cache_dir, exist_ok=True)
  if ds_name not in movielens_config.ML_DATASETS:
    raise AssertionError(
      f"Unknown Movielens dataset '{ds_name}'. Choose one of {list(movielens_config.ML_DATASETS)}"
    )

  record = movielens_config.ML_DATASETS[ds_name]
  checksum = record.get("checksum")
  data_file = os.path.basename(record["file"])
  data_path = tf.keras.utils.get_file(
    data_file,
    record["file"],
    file_hash=checksum,
    cache_dir=cache_dir,
    cache_subdir=".")
  return data_path


def prepare_datasets(filepath):
  ml = MovieLens(filepath)
  df = ml.full_table
  df.drop(["item"], axis=1, inplace=True)

  target = "rating"
  numerical = [
    "user_ratings_mean",
    "user_ratings_count",
    "item_ratings_mean",
    "item_ratings_count",
    "release year",
  ]
  onehot = set(df.columns) - set(numerical + [target])

  # Scale target to [0, 1]
  target_min = df[target].min()
  target_max = df[target].max()
  df[target] = (df[target] - target_min) / (target_max - target_min)

  # Scale numerical
  for n in numerical:
    df[n] = (df[n] - df[n].mean()) / df[n].std()

  # Scale onehot.
  for o in onehot:
    df[o] -= 0.5

  # Split per-user
  datasets = []
  for _, v in df.groupby("user"):
    rating = v.pop("rating")
    datasets.append((v.values, rating.values))
  datasets = np.asarray(datasets, dtype=object)
  return datasets, target_min, target_max


def take_n_random(n, xy, rng=np.random.default_rng()):
  idx = rng.integers(0, len(xy[0]), n)
  return xy[0][idx], xy[1][idx]


def prepare_dataset(filepath,
                    ratings_per_user,
                    users_per_batch,
                    rng=np.random.default_rng()):
  datasets, target_min, target_max = prepare_datasets(filepath)
  n_users = len(datasets)
  n_features = datasets[0][0].shape[1]

  if users_per_batch > n_users:
    logging.warning(
      "Users per batch %d is larger than total number of users in the dataset %d",
      users_per_batch, n_users)
    logging.warning("Setting users per batch to %d", n_users)
    users_per_batch = n_users

  def generator():
    for batch in utils.roundrobin_generator(datasets, users_per_batch):
      xy = [take_n_random(ratings_per_user, ds) for ds in batch]
      x = tf.stack([d[0] for d in xy])
      y = tf.stack([d[1] for d in xy])

      yield (x, y), y

  dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
      (tf.TensorSpec(
        shape=(users_per_batch, ratings_per_user, n_features),
        dtype=tf.float64),
       tf.TensorSpec(
         shape=(users_per_batch, ratings_per_user), dtype=tf.float32)),
      tf.TensorSpec(
        shape=(users_per_batch, ratings_per_user), dtype=tf.float32),
    ))

  return dataset, n_users, users_per_batch, target_min, target_max
