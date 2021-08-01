from absl import flags
from sklearn import model_selection

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
  X, y = data.get_numpy()

  X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, stratify=y
  )

  m = model.get_model()

  # specify parameters via map, definition are same as c++ version
  params = {
    'n_estimators': 3000,
    'objective': 'binary:logistic',
    'learning_rate': 0.01,
    #'gamma':0.1,
    #'subsample':0.8,
    #'colsample_bytree':0.3,
    #'min_child_weight':3,
    'max_depth': 3,
    #'seed':1024,
    'n_jobs': -1
  }
  m.set_params(**params)
  m.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    early_stopping_rounds=100,
    verbose=True,
  )
