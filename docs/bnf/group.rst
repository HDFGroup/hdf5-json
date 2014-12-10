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
   link: "{"
       : "title" ":" `link_name` ","
       : "href" ":" `path` | `fragment` | `url` ","
       : "creationProperties" ":" `lcpl`
       : "}"
   lcpl: "{"
       : "nameCharEncoding" ":" `char_encoding`
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
