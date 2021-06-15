import functools
import os
import zipfile

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

  def read_csv(self, file_names, column_names):
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
            names=column_names,
            header=header,
            engine=engine,
            encoding="latin1")

        # Keep only clean data.
        df.dropna(inplace=True)
        return df

  @functools.cached_property
  def ratings(self) -> pd.DataFrame:
    return self.read_csv(
        self.ratings_files,
        column_names=("user", "item", "rating", "timestamp"))

  @functools.cached_property
  def items(self, column_names=("item", "title", "genres")) -> pd.DataFrame:
    df = self.read_csv(self.items_files, column_names)

    # Add year column (extracted from the title)
    df["Release Date"] = df["title"].map(
        lambda x: pd.to_numeric(x.strip()[-5:-1], errors="coerce"))

    # Drop title since we do not use it.
    df.drop(["title"], axis=1)

    # Convert genres to dummy variables.
    genres = df.pop("genres")
    genres = pd.get_dummies(
        genres.map(lambda x: x.split("|")).explode()).sum(level=0)
    df = pd.concat([df, genres], axis=1)

    # Keep only clean data.
    df.dropna(inplace=True)
    return df

  @functools.cached_property
  def full_table(self) -> pd.DataFrame:
    item_means = self.ratings.groupby("item").rating.agg({
        "mean",
        "count",
    }).rename(columns={
        "mean": "item_ratings_mean",
        "count": "item_ratings_count",
    })

    user_means = self.ratings.groupby("user").rating.agg({
        "mean",
        "count",
    }).rename(columns={
        "mean": "user_ratings_mean",
        "count": "user_ratings_count",
    })

    # Join ratings with user/item aggregates.
    df = self.ratings\
      .merge(user_means, how="left", on="user")\
      .merge(item_means, how="left", on="item")\
      .merge(self.items, how="left", on="item")

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
