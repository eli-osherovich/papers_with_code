import pathlib
from typing import Union

import gin

from ...arxiv_1711_09784 import tree

GIN_CONFIG_FILE = 'tree_model_config.gin'


@gin.configurable
def get_tree_model(depth: int,
                   n_classes: int) -> Union[tree.LeafNode, tree.InnerNode]:
  return tree.TreeModel(n_classes=n_classes, depth=depth)


gin.parse_config_file(pathlib.Path(__file__).parent.resolve() / GIN_CONFIG_FILE)
