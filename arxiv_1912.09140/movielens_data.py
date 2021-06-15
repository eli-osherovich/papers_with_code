import functools
import os
import zipfile

import numpy as np
import pandas as pd
from tensorflow import keras

DATASETS = {
    "100k": {
        "file": "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "checksum": "0e33842e24a9c977be4e0107933c0723",
    },
    "1m": {
        "file": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "checksum": "c4d9eecfca2ab87c1945afe126590906",
    },
    "10m": {
        "file": "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "checksum": "ce571fd55effeba0271552578f2648bd",
    },
    "20m": {
        "file": "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "checksum": "cd245b17a1ae2cc31bb14903e1204af3",
    },
    "25m": {
        "file": "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "checksum": "6b51fb2759a8657d3bfcbfc42b592ada",
    },
    "latest-small": {
        "file":
            "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    },
    "latest": {
        "file": "http://files.grouplens.org/datasets/movielens/ml-latest.zip",
    }
}


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
  if ds_name not in DATASETS:
    raise AssertionError(
        f"Unknown Movielens dataset '{ds_name}'. Choose one of {list(DATASETS.keys())}"
    )

  record = DATASETS[ds_name]
  checksum = record.get("checksum")
  data_file = os.path.basename(record["file"])
  data_path = keras.utils.get_file(
      data_file,
      record["file"],
      file_hash=checksum,
      cache_dir=cache_dir,
      cache_subdir=ds_name)
  return data_path
