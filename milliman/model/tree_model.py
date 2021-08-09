from this import d
from typing import Union

from ...arxiv_1711_09784 import tree


def get_tree_model(depth: int) -> Union[tree.LeafNode, tree.InnerNode]:
  return tree.TreeModel(n_classes=1, depth=depth)
