"""A bunch of useful types"""

from typing import Union, Callable, List

import numpy as np
import tensorflow as tf

Number = Union[float, int, np.float16, np.float32, np.float64, np.int8,
               np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
               np.uint64,]

Initializer = Union[None, dict, str, Callable,
                    tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable,
                    tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[List[Union[Number, list]], tuple, Number, np.ndarray,
                   tf.Tensor, tf.SparseTensor, tf.Variable]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]
