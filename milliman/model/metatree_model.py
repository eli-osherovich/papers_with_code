import functools

import gin
import tensorflow as tf

from . import metatree

_B_LO = tf.constant([
  -2.2338567283409025, -3.321779081441261, -2.7279813295174837,
  -4.773952579363098, -3.248870315174375, -1.3781703984907252,
  -0.9421637820708985, -1.053413018870969, -3.5943817696146394,
  -4.572632218970465, -2.29128784747792, -1.2089410496539776,
  -1.2375966910186262, -1.0672569667868559, -4.096068575814836,
  -1.370291152174099, -3.021023382643871, -6.591239977761729,
  -11.50362261782493, -0.4823542675723937, -2.4784787961282104,
  -4.41733183186644, -0.6352234031660235, -2.1343747458109497,
  -2.881375640452898, -7.000000000000001, -2.015760685012114,
  -0.752035723846475, -2.440598643597598
])

_B_HI = tf.constant([
  1.4584684424705066, 1.41999070088631, 1.9129225803474537, 1.0904665413522914,
  0.6654312693730648, 1.5697876731471898, 1.727485659452309, 1.6476460038751053,
  1.4788138851768238, 0.7187454693270836, 0.4364357804719845, 0.827170191868511,
  0.8080176743014172, 0.9369814684936247, 0.24413653763134782,
  0.7297719162919529, 0.33101365773767794, 0.15171652122725207,
  0.08692913816996169, 2.0731650308243945, 0.4034732923929645,
  0.22638100058185434, 1.5742493034984064, 0.46852128566581813,
  0.3470564496904051, 0.14285714285714288, 0.4960906358752555,
  1.3297240653479196, 0.409735538706166
])


@gin.configurable
def get_model(depth: int, input_dim: int, emb_dim: int) -> tf.keras.Model:
  encoder_fn = functools.partial(
    metatree.gen_input_encoder, input_dim=input_dim, emb_dim=emb_dim
  )
  inner_model_fn = functools.partial(
    metatree.gen_inner_model,
    input_dim=input_dim,
    emb_dim=emb_dim,
    b_limits=(_B_LO, _B_HI)
  )
  leaf_model_fn = functools.partial(
    metatree.gen_leaf_model, input_dim=input_dim, emb_dim=emb_dim
  )

  model = metatree.TreeModel(
    depth=depth,
    encoder_model_fn=encoder_fn,
    inner_model_fn=inner_model_fn,
    leaf_model_fn=leaf_model_fn
  )
  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
      tf.keras.metrics.BinaryAccuracy(name="acc"),
    ],
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, amsgrad=True),
  )

  return model
