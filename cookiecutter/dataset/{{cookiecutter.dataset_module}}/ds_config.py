"""{{cookiecutter.dataset}} dataset schema.
"""
from absl import logging
import numpy as np
import pandas as pd

_INT_TYPE = np.int32
_FLOAT_TYPE = np.float32
_BOOL_TYPE = np.int32
_CATEGORICAL_TYPE = pd.CategoricalDtype

logging.error("Please review the schema below and remove this exception")
feature_dict = {
  # Example:
  #
  # "col1": _INT_TYPE,
  # "col2": CATEGORICAL_TYPE([
  #   "cat",
  #   "dog",
  #   "human",
  #   "unknown",
  # ])
  {%- macro string_to_quoted_list(string) -%}
  "{{ string.split(",")|join('", "') }}"
  {%- endmacro -%}

  {%- macro type_mapper(type, categories_string="") -%}
  {%if type == "float" %}_FLOAT_TYPE{% endif -%}
  {%if type == "binary" %}_BOOL_TYPE{% endif -%}
  {%if type == "int" %}_INT_TYPE{% endif -%}
  {%if type == "categorical" -%}
  _CATEGORICAL_TYPE([{{ string_to_quoted_list(categories_string) }}])
  {%- endif -%}
  {% endmacro -%}

  {%- macro column_schema(prefix, number, type="float", categories_string="") -%}
  "{{prefix}}{{number}}": {{type_mapper(type, categories_string)}},
  {%- endmacro -%}

  {% set target_columns = cookiecutter.target_columns.split(",") %}
  {% for n in range(cookiecutter.num_columns|int - target_columns|length) %}
  {{column_schema("feature", n)}}
  {%- endfor -%}
  {% for t in target_columns %}
  {{column_schema(t, "", cookiecutter.target_type, cookiecutter.categories)}}
  {%- endfor %}
}
