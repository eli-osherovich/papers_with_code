import gin
import tensorflow as tf

@gin.configurable
def get_model() -> tf.keras.Model:
  return  {{cookiecutter.model}}()

class {{cookiecutter.model}}(tf.keras.Model):
  def __init__(self, **kwargs) -> None:
      super().__init__(**kwargs)
