"""{{cookiecutter.dataset}} dataset config.
"""

from .. import dataset
from .. import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

{%- macro split_config(name, uri, checksum="FIXME: add real checksum") -%}
{%- if uri -%}
"{{name}}":
    dataset.DatasetFile(
      uri=utils.make_uri("{{uri}}"),
      checksum="{{checksum}}",
    ),
{%- endif -%}
{%- endmacro %}


SPLITS = {
  {{split_config("train", cookiecutter.train_file_uri)}}
  {{split_config("val", cookiecutter.val_file_uri)}}
  {{split_config("test", cookiecutter.test_file_uri)}}
}
