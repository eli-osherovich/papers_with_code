from typing import Union

import gin

from ...arxiv_1711_09784 import tree


@gin.configurable
def get_tree_model(depth: int,
                   n_classes: int) -> Union[tree.LeafNode, tree.InnerNode]:
  return tree.TreeModel(n_classes=n_classes, depth=depth)
