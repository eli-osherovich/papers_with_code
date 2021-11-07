"""Training/tuning routines for {{cookiecutter.model}}.

"""
import gin


@gin.configurable
def train(X, y):
  raise NotImplementedError()


@gin.configurable
def train_cv(X, y):
  raise NotImplementedError()


@gin.configurable
def tune(X, y):
  raise NotImplementedError()
