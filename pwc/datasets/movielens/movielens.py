import re

import numpy as np
import pandas as pd

from .. import dataset
from .. import io


class Movielens(dataset.Dataset):
  """Base class for all Movielens variants"""

  _TARGET_COLUMN = "rating"

  _ENCODING = "utf-8"
  _HEADER = 0
  _ITEMS_FILE = "movies"
  _RATINGS_FILE = "ratings"
  _SEP = ","
  _USERS_FILE = None

  def __init__(self, flavor: str) -> None:
    super().__init__()
    self._flavor = flavor

  @property
  def ds_path(self):
    return self.download_dataset(self._flavor)

  def _generate_dataframe(self, split_name):
    if split_name != self._flavor:
      raise AssertionError(
        f"split_name must be `{self._flavor}`, called with `{split_name}`"
      )

    ratings = self.ratings()
    items = self.items()
    items = self.items_postprocess(items)
    ratings = ratings.merge(items, how="inner", on="item").dropna()
    y = ratings.pop(self._TARGET_COLUMN)
    return ratings, y

  def items_postprocess(self, items):
    # Extract release year.
    def extract_year(title):
      m = re.search(r"(\d{4})\)*\s*$", title)
      if m:
        return m.group(1)

    # we do not use title
    title = items.pop("title")
    items["release year"] = pd.to_numeric(
      title.apply(extract_year), errors="coerce", downcast="float"
    )

    # Make sure all datasets have the same set of columns.
    for col in ["IMAX", "unknown"]:
      if col not in items:
        items[col] = self.feature_dict[col](0)

    return items

  def ratings(self) -> pd.DataFrame:
    df = self._load_file(
      self._RATINGS_FILE,
      ["user", "item", "rating", "timestamp"],
      is_re=True,
      sep=self._SEP,
      usecols=("user", "item", "rating"),
    )

    # Add aggregate values: 'mean', 'count' for each item.
    # The aggregates are done for both 'user' and 'item'
    for agg_key in ["user", "item"]:
      agg_data = (
        df.groupby(agg_key).rating.agg(["mean", "count"]).rename(
          columns={
            "mean": f"{agg_key}_ratings_mean",
            "count": f"{agg_key}_ratings_count",
          }
        ).astype(np.float32)
      )
      df = df.merge(agg_data, how="inner", on=agg_key)

      # Avoid data leakage: remove the current rating from aggregated data
      # We use the following formula:
      # (x*n - y)/(n-1) = x + (x-y)/(n-1)
      df[f"{agg_key}_ratings_count"] -= 1
      df[f"{agg_key}_ratings_mean"] += (
        df[f"{agg_key}_ratings_mean"] - df["rating"]
      ) / df[f"{agg_key}_ratings_count"]

    return df

  def items(self) -> pd.DataFrame:
    df = self._load_file(
      self._ITEMS_FILE,
      ["item", "title", "genres"],
      is_re=True,
      sep=self._SEP,
      encoding=self._ENCODING,
    )

    # Coerce same features despite minor diffs:
    # 'unknown' and '(no genres listed)'.
    # "Children's" and "Children"
    df["genres"] = df["genres"].str.replace(
      "(no genres listed)", "unknown", regex=False
    )
    df["genres"] = df["genres"].str.replace(
      "Children's", "Children", regex=False
    )

    # Convert genres to dummies.
    genres = df.pop("genres")
    genres = (
      pd.get_dummies(genres.map(lambda x: x.split("|")).explode()
                    ).groupby(level=0).sum()
    )
    df = pd.concat([df, genres], axis=1)

    return df

  def users(self) -> pd.DataFrame:
    return pd.DataFrame()

  def _load_file(self, name, features, *, is_re=False, **kwargs):
    file_acc = io.FileAccessor(self.ds_path, name=name, is_re=is_re)
    feature_dict = {f: self.feature_dict[f] for f in features}
    file_reader = io.PandasCSVReader(
      header=self._HEADER, names=feature_dict, dtype=feature_dict, **kwargs
    )
    return file_acc.read(file_reader)


class MovielensLatestSmall(Movielens):
  """Movielens latest-small dataset"""

  def __init__(self) -> None:
    super().__init__("latest-small")


class MovielensLatest(Movielens):
  """Movielens latest dataset"""

  def __init__(self) -> None:
    super().__init__("latest")


class Movielens25M(Movielens):
  """Movielens 25m dataset"""

  def __init__(self) -> None:
    super().__init__("25m")


class Movielens20M(Movielens):
  """Movielens 20m dataset"""

  def __init__(self) -> None:
    super().__init__("20m")


class Movielens10M(Movielens):
  """Movielens 10m dataset"""

  _HEADER = None
  _SEP = "::"

  def __init__(self) -> None:
    super().__init__("10m")


class Movielens1M(Movielens):
  """Movielense 1M dataset"""

  _ENCODING = "iso-8859-1"
  _HEADER = None
  _SEP = "::"

  def __init__(self) -> None:
    super().__init__("1m")

  def users(self) -> pd.DataFrame:
    return self._load_file(
      "ml-1m/users.dat",
      ["user", "gender", "age", "occupation_num", "zipcode"],
      sep=self._SEP,
    )


class Movielens100K(Movielens):
  """Movielens 100k dataset."""

  _HEADER = None
  _RATINGS_FILE = "u.data"
  _SEP = "\t"

  def __init__(self) -> None:
    super().__init__("100k")

  def users(self) -> pd.DataFrame:
    return self._load_file(
      "ml-100k/u.user",
      ["user", "item", "gender", "occupation", "zipcode"],
      sep="|",
    )

  def items(self) -> pd.DataFrame:
    return self._load_file(
      "ml-100k/u.item",
      [
        "item",
        "title",
        "release date",
        "video release date",
        "IMDb URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
      ],
      sep="|",
      encoding="iso-8859-1",
      usecols=[
        "item",
        "title",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
      ],
    )
