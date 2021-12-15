from glob import glob
from os import path
import pathlib
from typing import Union

import joblib
import numpy as np
import pandas as pd

DataType = Union[dict[str, pd.DataFrame], pd.DataFrame]
PathType = Union[str, pathlib.Path]


def get_feature_importance_file(
  data_type: str, features_type: str
) -> pathlib.Path:
  dir_path = pathlib.Path(__file__).parent / "features"
  if features_type:
    return dir_path / f"{data_type}_{features_type}_features_importance.pkl"
  else:
    return dir_path / f"{data_type}_features_importance.pkl"


def get_model_file(data_type: str, features_type: str) -> pathlib.Path:
  dir_path = pathlib.Path(__file__).parent / "models"
  if features_type:
    return dir_path / f"{data_type}_{features_type}_model.pkl"
  else:
    return dir_path / f"{data_type}_model.pkl"


def get_df_files(data_type: str, features_type: str) -> list[pathlib.Path]:
  dir_path = pathlib.Path(__file__).parent / "data"
  if features_type:
    pattern = f"{data_type}_{features_type}_*.feather"
  else:
    pattern = f"{data_type}.feather"
  return list(dir_path.glob(pattern))


def get_important_features(
  data_type: str,
  features_type: str,
  imp_thresh: float,
  *,
  logger=None
) -> Union[None, list[str]]:
  feats_file = get_feature_importance_file(data_type, features_type)
  if feats_file.exists():
    feats = joblib.load(feats_file)
    feats /= feats.max()
    n_feats = len(feats)
    feats = feats[feats > imp_thresh]
    n_imp_feats = len(feats)
    res = []
    for f in feats.index:
      f = f.removeprefix("ord__")
      res.append(f)
    res.append("SK_ID_CURR")

    if logger:
      logger.info(f"Keeping {n_imp_feats} out of {n_feats} from {feats_file}")
    return res
  else:
    if logger:
      logger.info(f"Feature importance file {feats_file} does not exist")
      return None


def remove_null_columns(
  data: DataType, thresh: float, *, logger=None
) -> DataType:

  def _remove_null_columns(df: pd.DataFrame):
    m, n = df.shape
    null_counts = df.isnull().sum()
    non_null_cols = null_counts[null_counts <= m * thresh].index
    df = df[non_null_cols]

    if logger:
      logger.info(
        f"Removed {n - df.shape[1]} columns that are mostly NULLs from {n} columns"
      )
    return df

  if isinstance(data, pd.DataFrame):
    return _remove_null_columns(data)
  else:
    res = {}
    for key, df in data.items():
      res[key] = _remove_null_columns(df)
    return res


def remove_high_corr(
  df: pd.DataFrame, corr_thresh: float, *, logger=None
) -> pd.DataFrame:
  # Preserve special columns
  id_ = "SK_ID_CURR"
  target_ = "TARGET"
  id_col = df.pop(id_) if id_ in df else None
  target_col = df.pop(target_) if target_ in df else None

  num_df = df.select_dtypes(include=["number", "bool"])
  corr_mat = np.corrcoef(
    num_df.dropna().to_numpy(dtype=np.float32), rowvar=False
  )
  corr_mat = np.abs(np.triu(corr_mat, k=1))
  corr_cols = num_df.columns[(corr_mat > corr_thresh).any(axis=0)]

  if logger:
    logger.info(
      f"Dropping {len(corr_cols)} features out of {len(corr_mat)} due to high correlation"
    )

  df = df.drop(corr_cols, axis=1)
  if id_col is not None:
    df[id_] = id_col
  if target_col is not None:
    df[target_] = target_col

  return df


def load_features_data(
  data_type: str,
  features_type: str,
  *,
  cols=None,
  logger=None
) -> pd.DataFrame:
  dfs = []
  for f_name in get_df_files(data_type, features_type):
    df = pd.read_feather(f_name, columns=cols)
    dfs.append(df)
  df = pd.concat(dfs, ignore_index=True)

  if logger:
    logger.info(f"Loaded {data_type}_{features_type} features: {df.shape}")
  return df


def save_model(model, data_type: str, features_type: str) -> None:
  model_path = get_model_file(data_type, features_type)
  features_path = get_feature_importance_file(data_type, features_type)
  joblib.dump(model, model_path)
  joblib.dump(
    model.levels[0][0].ml_algos[0].get_features_score(), features_path
  )


def dump_feather(data: dict[str, pd.DataFrame]):
  data_dir = pathlib.Path(__file__).parent / "data"
  data_dir.mkdir(exist_ok=True)

  for key, df in data.items():
    df.to_feather(data_dir / f"{key}.feather")
