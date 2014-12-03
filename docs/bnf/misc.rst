
.. productionlist::
   ocp: "attributeCreationOrder" ":" `attr_crt_order` ","
      : "attributePhaseChange" ":" `attr_phase_change` ","
      : "trackTimes" ":" `track_times`
   attr_crt_order: "H5P_CRT_ORDER_TRACKED" | "H5P_CRT_ORDER_INDEXED"
   attr_phase_change: "{"
                    : "maxCompact" ":" `non_negative_integer` ","
		    : "minDense" ":" `non_negative_integer`
		    : "}"
   track_times: "false" | "true"


.. productionlist::
   byte_array: **TBD**
   dims_array: positive_integer_array
   maxdims_array: **TBD**
   non_negative_integer: **TBD**
   non_negative_integer_array: **TBD**
   positive_integer: **TBD**
   positive_integer_array: **TBD**
   utc_datetime: **TBD**


.. productionlist::
   ascii_string: **TBD**
   char_string: **TBD**
   utf8_string: **TBD**
   hdf5_path_name: **TBD**
   hdf5_path_name_list: **TBD**
   url: **TBD**
   url_fragment: **TBD**
   url_path: **TBD**
