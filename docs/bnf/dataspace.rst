Dataspace
=========

.. productionlist::
   dataspace: `null_dataspace` | `scalar_dataspace` | `simple_dataspace`
   null_dataspace: "{"
	  : "class" ":" "H5S_NULL"
	  : "}"
   scalar_dataspace: "{"
	  : "class" ":" "H5S_SCALAR"
	  : "}"
   simple_dataspace: "{"
	  : "class" ":" "H5S_SIMPLE" ","
          : "dims" ":" `dims_array` ","
          : "maxdims" ":" `maxdims_array`
	  : "}"


.. productionlist::
   dataspace_selection: "{"
                      : "class" ":" `selection_class` ","
		      : "selection" ":" ( `hyperslab_selection`
		      :                  | `point_selection`)
   selection_class:  "H5S_SEL_HYPERSLABS"
                  :| "H5S_SEL_POINTS"
   hyperslab_selection: "[" `block_list` "]"
   block_list: `block` ("," `block`)*
   block: "{"
        : "start" ":" `coordinate` ","
	: "opposite" ":" `coordinate`
	: "}"
   point_selection: "[" `coordinate_list` "]"
   coordinate_list: `coordinate` ("," `coordinate`)*
   coordinate: `non_negative_integer_array`
