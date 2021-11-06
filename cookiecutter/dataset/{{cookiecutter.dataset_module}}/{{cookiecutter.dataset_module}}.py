{%- macro string_to_quoted_list(string) -%}
"{{ string.split(",")|join('", "') }}"
{%- endmacro -%}
"""Implementation of the {{cookiecutter.dataset}} dataset.
"""
import pandas as pd

from .. import dataset


class {{cookiecutter.dataset_class}}(dataset.Dataset):
  """{{cookiecutter.dataset}} dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=[{{string_to_quoted_list(cookiecutter.target_columns)}}], df_args={"skiprows": {{cookiecutter.skip_n_rows}}})
