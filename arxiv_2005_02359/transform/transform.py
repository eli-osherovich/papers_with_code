from typing import Optional

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  'n_transforms',
  default=64,
  help='Number of transforms to use',
  lower_bound=1,
)

flags.DEFINE_integer(
  'transform_dim',
  default=64,
  help='Dimensionality of the transformed data',
  lower_bound=1,
)


def get_transforms(x_dim: int, seed: Optional[int] = None) -> tf.Tensor:
  return tf.random.normal((FLAGS.n_transforms, FLAGS.transform_dim, x_dim),
                          seed=seed)


@tf.function
def apply_transforms(trans: tf.Tensor,
                     batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
  batch_size = batch.shape[0]
  if batch_size is None:
    raise RuntimeError('Batch size must be constant, use drop_remainder=True')

  n_trans = trans.shape[0]
  trans_dim = trans.shape[1]

  batch_transformed = tf.einsum('tdn, bn -> tbd', trans, batch)
  batch_transformed = tf.reshape(batch_transformed,
                                 [batch_size * n_trans, trans_dim])
  labels = tf.repeat(tf.range(n_trans), batch_size)
  return batch_transformed, labels
