from typing import Optional, Union

import gin

from ...arxiv_1711_09784 import tree


@gin.configurable
def get_model(
  depth: int,
  n_classes: int,
  leaf_initializer: Optional[float] = None
) -> Union[tree.LeafNode, tree.InnerNode]:
  return tree.TreeModel(
    n_classes=n_classes, depth=depth, leaf_initializer=leaf_initializer)
