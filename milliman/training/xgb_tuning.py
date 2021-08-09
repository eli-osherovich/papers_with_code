import gin

from .. import data


@gin.tunable
def tune(
  k_folds: int = 5,
  n_repeats: int = 20,
  patience: int = 100,
  random_state: int = 42
):
  X, y = data.get_numpy()

  # Count positive and negative examples
  n_pos = y.sum()
  n_neg = y.size - n_pos

  n_features = X.shape[-1]
  train_samples = int(X.shape[0] / k_folds * (k_folds - 1))


def xgb_tunable(config):
  pass


def tree_tunable(config):
  pass
