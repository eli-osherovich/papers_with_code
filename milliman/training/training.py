import numpy as np
import ray.tune
from absl import flags
from ray.tune.suggest.hyperopt import HyperOptSearch as search_alg
from sklearn import model_selection

from .. import data, model

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  'k_folds', 5, 'Number of k-folds in repeated CV.', lower_bound=2
)

flags.DEFINE_integer(
  'n_repeats', 20, 'Number of repeats in repeated CV.', lower_bound=1
)

flags.DEFINE_integer(
  'patience', 1000, 'Early stopping patience.', lower_bound=1
)

flags.DEFINE_integer(
  'cv_random_state', 42, 'CV splitter random_state', lower_bound=0
)

flags.DEFINE_string('save_dir', './saved_model', 'Model save directory.')


def train():
  # It seems that FLAGs are not pickable...
  k_folds = FLAGS.k_folds
  n_repeats = FLAGS.n_repeats
  cv_random_state = FLAGS.cv_random_state
  patience = FLAGS.patience

  X, y = data.get_numpy()

  n_pos = y.sum()
  n_neg = y.size - n_pos

  n_features = X.shape[-1]
  train_samples = int(X.shape[0] / k_folds * (k_folds - 1))

  tunable_params = {
    # 'alpha': ray.tune.uniform(0, 1),

    # The number of columns is an integer, hence,
    # we do not need to sample continuum. Note that the expression
    # of the type round(x / (1/y)) * (1/y) creates x' that is divisible by y.
    'colsample_bytree':
      ray.tune.quniform(
        round(0.2 * n_features) / n_features, 1, 1 / n_features
      ),
    # 'gamma':
    #   ray.tune.uniform(0, 1),
    # 'lambda':
    #   ray.tune.uniform(0, 2),
    'learning_rate':
      ray.tune.loguniform(1e-3, 1e-1),
    'max_depth':
      ray.tune.randint(4, 10),
    'min_child_weight':
      ray.tune.choice([1, 2, 3, 4, 5, 6]),
    'scale_pos_weight':
      ray.tune.uniform(1, n_neg / n_pos),

    # The number of training samples in an integer, hence,
    # we do not need to sample continuum. Note that the expression
    # of the type round(x / (1/y)) * (1/y) creates x' that is divisible by y.
    'subsample':
      ray.tune.quniform(
        round(0.5 * train_samples) / train_samples, 1.0, 1 / train_samples
      ),
  }

  X, y = data.get_numpy()
  m = model.get_model()
  m.set_params(n_estimators=100_000, objective='binary:logistic', n_jobs=1)

  def tuneable(config):
    m.set_params(**config)
    cv_splitter = model_selection.RepeatedStratifiedKFold(
      n_splits=k_folds, n_repeats=n_repeats, random_state=cv_random_state
    )
    res = []
    for train_index, test_index in cv_splitter.split(X, y, y):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      m.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        early_stopping_rounds=patience,
        verbose=False,
      )
      res.append(m.best_score)
    return {
      'auc_mean': np.mean(res),
      'auc_std': np.std(res),
      'auc_min': np.min(res),
      'auc_max': np.max(res),
    }

  ray.init()
  analysis = ray.tune.run(
    tuneable,
    config=tunable_params,
    metric='auc_mean',
    mode='max',
    num_samples=5000,
    search_alg=search_alg(metric='auc_mean', mode='max'),
    fail_fast=True,
  )
  print('Best result: ', analysis.best_result)
