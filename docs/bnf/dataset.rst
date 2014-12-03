.. rubric:: Dataset


.. productionlist::
   dataset: "{"
          : "id" ":" `identifier` ","
	  : "alias" ":" ( `hdf5_path_name` | `hdf5_path_name_list` ) ","
	  : "attributes" ":" `attribute_collection` ","
	  : "type" ":" `datatype` | `datatype_object_ref` ","
	  : "shape" ":" `dataspace` ","
	  : "value" ":" `json_value` ","
	  : "creationProperties" ":" `dcpl`
	  : "}"


.. productionlist::
   dcpl: "{"
      : "allocTime" ":" `alloc_time` ","
      : "attributeCreationOrder" ":" `attr_crt_order` ","
      : "attributePhaseChange" ":" `attr_phase_change` ","
      : "fillTime" ":" `fill_time` ","
      : "fillValue" ":" `fill_value` ","
      : "filters" ":" "[" `filter_list` "]" ","
      : "layout" ":" `layout` ","
      : "trackTimes" ":" `track_times`
      : "}"
   alloc_time: "H5D_ALLOC_TIME_DEFAULT"
             :| "H5D_ALLOC_TIME_EARLY"
	     :| "H5D_ALLOC_TIME_INCR"
	     :| "H5D_ALLOC_TIME_LATE"
   fill_time: "H5D_FILL_TIME_IFSET"
            :| "H5D_FILL_TIME_ALLOC"
	    :| "H5D_FILL_TIME_NEVER"
   fill_value: `json_value`
   layout: `chunked`
         :| `external`
         :| "H5D_COMPACT"
	 :| "H5D_CONTIGUOUS"
   external: "[" `file_extent_list` "]"
   file_extent_list: `file_extent` ("," `file_extent`)*
   file_extent: "{"
              : "name" ":" `ascii_string` ","
	      : "offset" ":" `non_negative_integer`
	      : "size" ":" `positive_integer`
	      : "}"
   chunked: "{"
          : "class" ":" "H5D_CHUNKED"
          : "dims" ":" `dims_array`
          : "}"
