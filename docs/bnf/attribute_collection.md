# Attribute

```{eval-rst}
.. productionlist::
   attribute_collection: "[" `attribute_list` "]"
   attribute_list: `attribute` ("," `attribute`)*
   attribute: "{"
            : "name" ":" `ascii_string` | `unicode_string` ","
            : "type" ":" `datatype` | `datatype_ref` ","
            : "shape" ":" `dataspace` ","
            : "value" ":" `json_value` ","
            : "creationProperties" ":" `acpl`
            : "}"
   acpl: "{"
       : "nameCharEncoding" ":" `char_encoding`
       : "}"
```
