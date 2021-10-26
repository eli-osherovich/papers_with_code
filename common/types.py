"""A bunch of useful types"""

from collections.abc import Sequence
from typing import Callable, List, Union

import numpy as np
import tensorflow as tf

Number = Union[float, int, np.float16, np.float32, np.float64, np.int8,
               np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
               np.uint64,]

Initializer = Union[dict, str, Callable, tf.keras.initializers.Initializer]
Regularizer = Union[None, str, dict, tf.keras.regularizers.Regularizer,
                    Callable,]
Constraint = Union[None, str, dict, tf.keras.constraints.Constraint, Callable]
Activation = Union[None, str, Callable]
Loss = Union[str, tf.keras.losses.Loss, Callable]
Metric = Union[None, str, Callable, tf.keras.metrics.Metric]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[List[Union[Number, list]], tuple, Number, np.ndarray,
                   tf.Tensor, tf.SparseTensor, tf.Variable]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]

Vector = Sequence[Number]
