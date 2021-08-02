from absl import flags
from ray import tune
from sklearn import model_selection
import numpy as np

from .. import data, model

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 100, 'Number of training epochs', lower_bound=1)
flags.DEFINE_enum('keep', 'best', ['last', 'best'], 'Which model to keep.')
flags.DEFINE_integer('patience', 10, 'Early stopping patience.', lower_bound=0)
flags.DEFINE_string(
  'monitor', 'val_accuracy', 'Early stopping quantity to monitor.'
)
flags.DEFINE_string('save_dir', './saved_model', 'Model save directory.')


def train():
  tunable_params = {
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'max_depth': tune.randint(1, 9),
    'min_child_weight': tune.choice([1, 2, 3]),
    'subsample': tune.uniform(0.5, 1.0),
    'colsample_bytree': tune.uniform(0.2, 1),
    #'gamma':0.1,
  }

  X, y = data.get_numpy()
  m = model.get_model()
  m.set_params(n_estimators=10000, objective='binary:logistic', n_jobs=1)

  def tuneable(config):

    m.set_params(**config)
    res = []
    for i in range(10):
      X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, stratify=y
      )

      m.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        early_stopping_rounds=1000,
        verbose=False,
      )
      res.append(m.best_score)
    return {
      'auc_mean': np.mean(res),
      'auc_std': np.std(res),
      'auc_min': np.min(res),
      'auc_max': np.max(res),
    }

  analysis = tune.run(
    tuneable,
    config=tunable_params,
    num_samples=100,
    mode='max',
    metric='auc_mean',
  )
  print('Best result: ', analysis.best_result)
