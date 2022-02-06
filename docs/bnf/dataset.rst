Dataset
=======

.. productionlist::
   dataset: `identifier` ":" "{"
          : "alias" ":" `hdf5_path_name_array`
	  : "attributes" ":" `attribute_collection` ","
	  : "type" ":" `datatype` | `datatype_object_ref` ","
	  : "shape" ":" `dataspace` ","
	  : "value" ":" `json_value` ","
	  : "creationProperties" ":" `dcpl` ","
      : "byteStreams" ":" `byte_stream_array`
	  : "}"
   json_value:  `json_string`
             :| `json_number`
             :| `json_array`
             :| `json_null`

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
   alloc_time:  "H5D_ALLOC_TIME_DEFAULT"
             :| "H5D_ALLOC_TIME_EARLY"
	     :| "H5D_ALLOC_TIME_INCR"
	     :| "H5D_ALLOC_TIME_LATE"
   fill_time:  "H5D_FILL_TIME_IFSET"
            :| "H5D_FILL_TIME_ALLOC"
	    :| "H5D_FILL_TIME_NEVER"
   fill_value: `json_value`
   layout: `chunked_layout` | `compact_layout` | `contiguous_layout`
   chunked_layout: "{"
          : "class" ":" "H5D_CHUNKED"
          : "dims" ":" `dims_array`
          : "}"
   compact_layout: "{"
          : "class" ":" "H5D_COMPACT"
          : "}"
   contiguous_layout: "{"
          : "class" ":" "H5D_CONTIGUOUS" ","
	  : "externalStorage" ":" `external`
          : "}"
   external: "[" `file_extent_list` "]"
   file_extent_list: `file_extent` ("," `file_extent`)*
   file_extent: "{"
              : "name" ":" `ascii_string` ","
	      : "offset" ":" `non_negative_integer`
	      : "size" ":" `positive_integer`
	      : "}"
    byte_stream_array: "[" `byte_stream_list` "]"
    byte_stream_list: `byte_stream`, ("," `byte_stream`)*
    byte_stream: "{"
        : "offset" ":" `non_negative_integer` ","
        : "size" ":" `non_negative_integer` ","
        : "uuid" ":" `uuid` ","
        : "cksum" ":" `checksum` ","
        : "dspace_anchor" ":" `dims_array`
        : "}"
    checksum: "{"
        : "type" ":" `identifier` ","
        : "value" ":" `ascii_string_wo_slash`
        : "}"
