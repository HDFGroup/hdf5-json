Miscellaneous
=============

.. productionlist::
   ocp: "attributeCreationOrder" ":" `attr_crt_order` ","
      : "attributePhaseChange" ":" `attr_phase_change` ","
      : "trackTimes" ":" `track_times`
   attr_crt_order:  "H5P_CRT_ORDER_TRACKED"
                 :| "H5P_CRT_ORDER_INDEXED"
   attr_phase_change: "{"
                    : "maxCompact" ":" `non_negative_integer` ","
		    : "minDense" ":" `non_negative_integer`
		    : "}"
   track_times: "false" | "true"

.. rubric:: URL

.. productionlist::
   url: `scheme` "://" `domain` [ ":" `port` ] "/" `path` [ "#" `fragment` ]
   scheme: "file" | "http"
   domain: `rfc1738_url_path`
   port: `non_negative_integer`
   path: ( "datasets" | "datatypes" | "groups" ) "/" `identifier`
   fragment: "h5(" `hdf5_path_name` ")"

.. productionlist::
   hdf5_path_name: [ "/" ] `link_name` ("/" `link_name` )*
   link_name: ascii_string_wo_slash | unicode_string_wo_slash


.. rubric:: Simple Types

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

.. rubric:: Date and Time

.. productionlist::
   utc_datetime: **TBD**

.. rubric:: Strings

.. productionlist::
   ascii_string_wo_slash: **TBD**
   ascii_string: **TBD**
   unicode_string_wo_slash: **TBD**
   unicode_string: **TBD**

.. rubric:: Identifier

.. productionlist::
   id_reference: `identifier`
   identifier: `uuid` | **TBD**
   uuid:  /[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/
