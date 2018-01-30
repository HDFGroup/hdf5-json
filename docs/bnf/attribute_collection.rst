Attribute
=========

.. productionlist::
   attribute_collection: "[" `attribute_list` "]"
   attribute_list: `attribute` ("," `attribute`)*
   attribute: "{"
            : "name" ":" `ascii_string` | `unicode_string` ","
            : "type" ":" ( `datatype` | "{" "idref" : `id_reference` "}" ) ","
            : "shape" ":" `dataspace` ","
            : "value" ":" `unicode_string` ","
            : "creationProperties" ":" `acpl`
            : "}"
   acpl: "{"
       : "nameCharEncoding" ":" `char_encoding`
       : "}"
