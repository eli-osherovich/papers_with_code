"""Training/tuning routines for TabNet.

"""
import gin


@gin.configurable
def train(X, y):
  return None
  raise NotImplementedError()


@gin.configurable
def train_cv(X, y):
  raise NotImplementedError()


@gin.configurable
def tune(X, y):
  raise NotImplementedError()
