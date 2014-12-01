.. rubric:: Dataspace


.. productionlist::
   dataspace: "H5S_NULL" | "H5S_SCALAR" | `simple_dataspace`
   simple_dataspace: "{"
	  : "class" ":" "H5S_SIMPLE" ","
          : "dims" ":" `dims_array` ","
          : "maxdims" ":" `maxdims_array`
	  : "}"


.. productionlist::
   dataspace_selection: `hyperslab_selection` | `point_selection`
   hyperslab_selection: "[" `block_list` "]"
   block_list: `block` ("," `block`)*
   block: "{"
        : "start" ":" `coordinate` ","
	: "opposite" ":" `coordinate`
	: "}"
   point_selection: "[" `coordinate_list` "]"
   coordinate_list: `coordinate` ("," `coordinate`)*
   coordinate: **TBD**
