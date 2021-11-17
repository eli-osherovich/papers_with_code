import collections

import numpy as np
import pytest

from . import utils


@pytest.mark.parametrize("batch_size", [-1.0, 0.0, 1.0, 1.5])
def test_batch_float(batch_size):
  with pytest.raises(AssertionError):
    rrg = utils.roundrobin_generator([1, 2, 3], batch_size)
    next(rrg)


@pytest.mark.parametrize("batch_size", [-2, -1, 0])
def test_batch_nonpositive(batch_size):
  with pytest.raises(AssertionError):
    rrg = utils.roundrobin_generator([1, 2, 3], batch_size)
    next(rrg)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 7])
def test_batch_size(batch_size):
  rrg = utils.roundrobin_generator([1, 2, 3], batch_size)
  for _ in range(10):
    assert len(next(rrg)) == batch_size


@pytest.mark.parametrize("array_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_unique_in_batch(array_len):
  arr = np.arange(array_len)
  for batch_size in np.arange(1, array_len + 1):
    rrg = utils.roundrobin_generator(arr, batch_size)
    for _ in range(1000 * array_len):
      batch = next(rrg)
      assert len(set(batch)) == batch_size


# Simulate TF's batch(drop_remainder=True): in this setup, each element from
# the array appears at most once.
@pytest.mark.parametrize("array_len", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_one_epoch_drop_remainder(array_len, batch_size):
  arr = np.arange(array_len)
  steps_per_epoch = np.floor(len(arr) / batch_size)
  rrg = utils.roundrobin_generator(arr, batch_size)

  counter = collections.Counter()
  for _ in np.arange(steps_per_epoch):
    counter.update(next(rrg))
  assert len(counter) == steps_per_epoch * batch_size
  assert np.all(np.array(list(counter.values())) == 1)


# Simulate several epochs such that every element form the array
# appears the same number of times. In this test we only evaluate
# cases where `batch_size` <= `array_len`.
@pytest.mark.parametrize("batch_size", [1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
@pytest.mark.parametrize("array_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_multiple_epochs(array_len, batch_size):
  M = 10

  arr = np.arange(array_len)
  rrg = utils.roundrobin_generator(arr, batch_size)

  # Lowest common multiple
  lcm = np.lcm(array_len, batch_size)

  # Greatest common divisor
  gcd = np.gcd(array_len, batch_size)

  # We process M*lcm elements which is divisible by both array_len and
  # batch_size.  Whenever array_len divisible by batch_size or vice versa, after
  # M*lcm steps each element must appear exactly M*lcm/array_len times.
  counter = collections.Counter()
  for _ in np.arange(M * lcm / batch_size):
    counter.update(next(rrg))
  assert len(counter) == array_len
  if gcd == batch_size or gcd == array_len:
    assert np.all(np.array(list(counter.values())) == M * lcm / array_len)
