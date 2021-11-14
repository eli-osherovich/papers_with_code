{%- macro string_to_quoted_list(string) -%}
"{{ string.split(",")|join('", "') }}"
{%- endmacro -%}
"""Implementation of the {{cookiecutter.dataset}} dataset.
"""

from .. import dataset
from ..typing import DatasetType


class {{cookiecutter.dataset_class}}(dataset.Dataset):
  """{{cookiecutter.dataset}} dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.{{cookiecutter.dataset_type|upper}}, target_columns=[{{string_to_quoted_list(cookiecutter.target_columns)}}])
