"""TabNet implementation in modern TF

  Reference:
      https://arxiv.org/abs/1908.07442
"""

from typing import Any, Optional

import tensorflow as tf
import tensorflow_addons as tfa

from ...common.types import Activation


def glu(x: tf.Tensor) -> tf.Tensor:
  """Generalized linear unit nonlinear activation."""
  half_dim = x.shape[-1] // 2
  return x[:, :half_dim] * tf.nn.sigmoid(x[:, half_dim:])


class TabNetBlock(tf.keras.Model):
  """Basic building block of TabNet.

  The basic block of the TabNet consists of the three layers:
  Dense -> BatchNorm -> Activation.
  An existing dense layer can be passed as an argument. This mechanism allows
  one to share a layer among several blocks. If not provided, a new one
  (non-shared) will be automatically created using `dense_args`
  arguments.

  Args:
      shared_dense (tf.keras.layers.Dense, optional): Shared dense layer. If not
        provided, the block with generate it using `dense_args` in this case,
        the layer will *not* be shared. Defaults to None.
      activation (Activation): Activation used at the block's output. Defaults
        to 'relu'.
      dense_args (dict, optional): Arguments for the dense layer. Used only if
        an external (shared) denslayer was not provided.
      batch_norm_args (dict, optional): Arguments to the BatchNorm layer.
      kwargs (dict, optional): Arguments passed to the parent class.
  """

  def __init__(
    self,
    *,
    shared_dense: Optional[tf.keras.layers.Dense] = None,
    activation: Activation = glu,
    skip_conn: bool = True,
    scale: float = 1.0,
    dense_args: dict[str, Any] = {},
    batch_norm_args: dict[str, Any] = {},
    **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.dense = shared_dense or tf.keras.layers.Dense(**dense_args)
    self.batch_norm = tf.keras.layers.BatchNormalization(**batch_norm_args)
    self.activation = tf.keras.layers.Activation(activation)
    self.skip_conn = skip_conn
    self.scale = scale
    # self.model will be created in build()

  def build(self, input_shape: tf.TensorShape) -> None:
    batch_size, *shape = input_shape
    x = tf.keras.Input(shape=shape, batch_size=batch_size)
    f = self.dense(x)
    f = self.batch_norm(f)
    f = self.activation(f)

    # All but the first blocks use skip connection.
    if self.skip_conn:
      f = (f + x) * self.scale

    self.model = tf.keras.Model(x, f)

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    return self.model(inputs, **kwargs)


class TabNetStep(tf.keras.Model):
  """FeatureTransformer of TabNet.


  Args:
      shared_dense1 (tf.keras.layers.Dense): First shared layer in every step.
      shared_dense2 (tf.keras.layers.Dense): Second shared layer in every step.
      scale (float, optional): Scaling coefficient used between the blocks.
        Defaults to sqrt(0.5),
      dense_args (dict, optional): Arguments for the non-shared dense layers.
      bn_args (dict, optional): Arguments for the BatchNorm layers.
  """

  _MASK_FEATURES = "mask_features"
  _DECISION_FEATURES = "decision_features"

  def __init__(
    self,
    *,
    shared_dense1: tf.keras.layers.Dense,
    shared_dense2: tf.keras.layers.Dense,
    mask_features_dim: int,
    decision_features_dim: int,
    relaxation_factor: float = 1.0,
    scale: float = 0.5**0.5,
    block_activation: Activation = glu,
    dense_args: dict = {},
    batch_norm_args: dict = {},
    **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.shared_dense1 = shared_dense1
    self.shared_dense2 = shared_dense2
    self.mask_features_dim = mask_features_dim
    self.decision_features_dim = decision_features_dim
    self.relaxation_factor = relaxation_factor
    self.scale = scale
    self.block_activation = block_activation
    self.dense_args = dense_args
    self.batch_norm_args = batch_norm_args
    # self.model will be created in build()

  def build(self, input_shape):
    transformer_input_shape, mask_shape, comp_mask_shape = input_shape
    del mask_shape

    self.feature_transformer_model = self._build_feature_transformer_model(
      transformer_input_shape
    )

    mask_features_shape = self.feature_transformer_model.output_shape[
      self._MASK_FEATURES]
    decision_features_shape = self.feature_transformer_model.output_shape[
      self._DECISION_FEATURES]

    self.decision_model = self._build_decision_model(decision_features_shape)

    # Mask model receives mask features calculated by a FeatureTransformer and
    # the aggregated complimentary mask.
    self.mask_model = self._build_mask_model(
      (mask_features_shape, comp_mask_shape)
    )

  def _build_feature_transformer_model(self, input_shape):
    # Input is a concatenation of the current mask and embedding.
    b1 = TabNetBlock(
      shared_dense=self.shared_dense1,
      activation=self.block_activation,
      skip_conn=False,
      scale=1.0,
      batch_norm_args=self.batch_norm_args
    )
    b2 = TabNetBlock(
      shared_dense=self.shared_dense2,
      activation=self.block_activation,
      skip_conn=True,
      scale=self.scale,
      batch_norm_args=self.batch_norm_args
    )
    b3 = TabNetBlock(
      activation=self.block_activation,
      skip_conn=True,
      scale=self.scale,
      dense_args=self.dense_args,
      batch_norm_args=self.batch_norm_args
    )
    b4 = TabNetBlock(
      activation=self.block_activation,
      skip_conn=True,
      scale=self.scale,
      dense_args=self.dense_args,
      batch_norm_args=self.batch_norm_args
    )
    batch_size, *shape = input_shape
    x = tf.keras.Input(shape=shape, batch_size=batch_size)
    f = b1(x)
    f = b2(f)
    f = b3(f)
    f = b4(f)

    # Feature transformer generates features for mask and decision calculations.
    mask_features = f[:, :self.mask_features_dim]
    decision_features = f[:, -self.decision_features_dim:]
    return tf.keras.Model(
      x,
      outputs={
        self._MASK_FEATURES: mask_features,
        self._DECISION_FEATURES: decision_features
      }
    )

  def _build_mask_model(self, input_shape):
    mask_features_shape, mask_shape = input_shape

    # not used -- must be (batch_size, self.mask_features_dim).
    del mask_features_shape

    batch_size, input_dim = mask_shape

    mask_features = tf.keras.Input(
      shape=(self.mask_features_dim,), batch_size=batch_size
    )
    complimentary_agg_mask_in = tf.keras.Input(
      shape=(input_dim,), batch_size=batch_size
    )
    m = tf.keras.layers.Dense(input_dim)(mask_features)
    m = tf.keras.layers.BatchNormalization()(m)
    m *= complimentary_agg_mask_in
    mask = tfa.activations.sparsemax(m)
    complimentary_agg_mask_out = complimentary_agg_mask_in * (
      self.relaxation_factor - mask
    )
    return tf.keras.Model([mask_features, complimentary_agg_mask_in],
                          [mask, complimentary_agg_mask_out])

  def _build_decision_model(self, input_shape):
    x = tf.keras.Input(input_shape[1:])
    decision = tf.keras.activations.relu(x)
    return tf.keras.Model(x, decision)

  def call(self, inputs, **kwargs):
    x, mask, complimentary_agg_mask_in = inputs
    masked_x = x * mask
    ft_res = self.feature_transformer_model(masked_x, **kwargs)
    decision = self.decision_model(ft_res[self._DECISION_FEATURES])
    new_mask, complimentary_agg_mask_out = self.mask_model(
      (ft_res[self._MASK_FEATURES], complimentary_agg_mask_in)
    )
    return x, decision, new_mask, complimentary_agg_mask_out
