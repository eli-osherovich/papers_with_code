"""{{cookiecutter.dataset}} download config.
"""
from .. import utils


{%- macro split_config(name, uri, checksum="FIXME: add real checksum") -%}
{%- if uri -%}
  "{{name}}": {
    "uri": utils.make_uri("{{uri}}"),
    "checksum": "{{checksum}}",
  },
{%- endif -%}
{%- endmacro %}

DATASETS = {
  {{split_config("train", cookiecutter.train_file_uri)}}
  {{split_config("val", cookiecutter.val_file_uri)}}
  {{split_config("test", cookiecutter.test_file_uri)}}
}
