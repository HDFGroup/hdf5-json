.. rubric:: Attribute


.. productionlist::
   attribute_collection: "attributes" ":" "[" `attribute_list` "]"
   attribute_list: `attribute` ("," `attribute`)*
   attribute: "{"
            : "name" ":" `ascii_string` | `utf8_string` ","
	    : "type" ":" `datatype` | `datatype_object_ref` ","
	    : "shape" ":" `dataspace` ","
	    : "value" ":" `json_value`
	    : "}"
