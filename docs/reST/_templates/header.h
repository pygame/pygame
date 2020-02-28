{#- Generate an extension module header file of doc strings #}

{%- set classmembers = ['attribute', 'method', 'staticmethod', 'classmethod'] %}

{%- macro cmacro(item) %}
{%-   if item['desctype'] in classmembers %}
{%-     set start_at = -2 %}
{%-   else %}
{%-     set start_at = 0 %}
{%-   endif %}
{%-   set name = item['fullname'] %}
{{-   'DOC_' + name.split('.')[start_at:]|join('')|replace('_', '')|upper }}
{%- endmacro %}

{%- macro join_sigs(item) %}
{%-   set sigs = item['signatures'] %}
{%-   if sigs %}
{{-     sigs|join('\\n') + '\\n' }}
{%-   else %}
{{-     '' }}
{%-   endif %}
{%- endmacro %}

{#- -#}


/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
{% for item in hdr_items -%}
#define {{ cmacro(item) }} "{{ join_sigs(item) }}{{ item['summary'] }}"
{% endfor %}

/* Docs in a comment... slightly easier to read. */

/*

{%  for item in hdr_items -%}
{{    item['fullname'] }}
{%    set sigs = item['signatures'] -%}
{%    if sigs -%}
{{      ' ' + sigs|join('\n ') }}
{%    endif -%}
{{    item['summary'] }}

{%  endfor -%}
*/
