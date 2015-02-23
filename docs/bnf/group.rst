Group
=====

.. productionlist::
   group: `identifier` ":" "{"
	: "attributes" ":" `attribute_collection` ","
	: "links" ":" `link_collection` ","
	: "created" ":" `utc_datetime` ","
	: "lastModified" ":" `utc_datetime` ","
	: "creationProperties" ":" `gcpl`
        : "}"

.. productionlist::
   link_collection: "[" `link_list` "]"
   link_list: `link` ("," `link`)*
   link: `hard_link` | `soft_link` | `external_link` | `ud_link`
   hard_link: "{"
       : "class" ":" "H5L_TYPE_HARD",
       : "title" ":" `link_name` ","
       : "collection" ":" ("datasets" | "datatypes" | "groups" )
       : "id" ":" `uuid`
       : "creationProperties" ":" `lcpl`
       : "}"
   soft_link: "{"
       : "class" ":" "H5L_TYPE_SOFT",
       : "title" ":" `link_name` ","
       : "h5path" ":" `unicode_string`
       : "creationProperties" ":" `lcpl`
       : "}"
   external_link: "{"
       : "class" ":" "H5L_TYPE_EXTERNAL",
       : "title" ":" `link_name` ","
       : "file" ":" `unicode_string`
       : "h5path" ":" `unicode_string`
       : "creationProperties" ":" `lcpl`
       : "}"
   ud_link: "{"
       : "class" ":" "H5L_TYPE_USER_DEFINED",
       : "title" ":" `link_name` ","
       : "target" ":" `byte_array`
       : "creationProperties" ":" `lcpl`
       : "}"
   lcpl: "{"
       : "charSet" ":" `char_encoding`
       : "creationOrder" ":" `non_negative_integer`
       : "}"

.. productionlist::
   gcpl: "{"
       : `ocp` ","
       : "filters" ":" "[" `link_name_filter_list` "]" ","
       : "linkCreationOrder" ":" `link_crt_order` ","
       : "linkPhaseChange" ":" `link_phase_change` ","
       : "linksEstimate" ":" `links_estimate` ","
       : "localHeapSizeHint" ":" `non_negative_integer` ","
       : "trackTimes" ":" `track_times`
       : "}"
   link_crt_order:  "H5P_CRT_ORDER_TRACKED"
                 :| "H5P_CRT_ORDER_INDEXED"
   link_phase_change: "{"
                    : "maxCompact" ":" `non_negative_integer` ","
		    : "minDense" ":" `non_negative_integer`
		    : "}"
   links_estimate: "{"
                 : "numEntries" ":" `non_negative_integer` ","
		 : "nameLength" ":" `non_negative_integer`
		 : "}"
   link_name_filter_list: `deflate_filter`
