
.. productionlist::
   ocp: "attributeCreationOrder" ":" `attr_crt_order` ","
      : "attributePhaseChange" ":" `attr_phase_change` ","
      : "trackTimes" ":" `track_times`
   attr_crt_order: "H5P_CRT_ORDER_TRACKED"
                 :| "H5P_CRT_ORDER_INDEXED"
   attr_phase_change: "{"
                    : "maxCompact" ":" `non_negative_integer` ","
		    : "minDense" ":" `non_negative_integer`
		    : "}"
   track_times: "false" | "true"


.. productionlist::
   byte_array: "[" `byte_list` "]"
   byte_list: `byte_value` ("," `byte_value`)*
   byte_value: /0x[0-F][0-F]/
   dims_array: positive_integer_array   
   maxdims_array: "[" `maxdims_list` "]"
   maxdims_list: `maxdim` ("," `maxdim`)*
   maxdim: `positive_integer` | "H5S_UNLIMITED"
   non_negative_integer_array: "[" `non_negative_integer_list` "]"
   non_negative_integer_list: `non_negative_integer`
                            : ("," `non_negative_integer`)*
   non_negative_integer: /integer >= 0/
   positive_integer_array: "[" `positive_integer_list` "]"
   positive_integer_list: `positive_integer`
                        : ("," `positive_integer`)*
   positive_integer: /integer > 0/


.. productionlist::
   utc_datetime: **TBD**


.. productionlist::
   ascii_string: **TBD**
   unicode_string: **TBD**
   hdf5_path_name: **TBD**
   hdf5_path_name_list: **TBD**
   url: **TBD**
   url_fragment: **TBD**
   url_path: **TBD**
