{{ name | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :ignore-module-all:
{% if modules %}

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
