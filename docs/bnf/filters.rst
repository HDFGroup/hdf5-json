Filter
======

.. productionlist::
   filter_list: `filter` ("," `filter`)*
   filter:  `deflate_filter`
         :| `fletcher32_filter`
	 :| `lzf_filter`
	 :| `nbit_filter`
	 :| `scaleoffset_filter`
	 :| `shuffle_filter`
	 :| `szip_filter`
	 :| `other_filter`

.. productionlist::
   deflate_filter: "{"
                 : "class" ":" "H5Z_FILTER_DEFLATE" ","
		 : "id" ":" 1 ","
		 : "level" ":" /[0-9]/
		 : "}"

.. productionlist::
   fletcher32_filter: "{"
                    : "class" ":" "H5Z_FILTER_FLETCHER32" ","
		    : "id" ":" 3
		    : "}"

.. productionlist::
   lzf_filter: "{"
              : "class" ":" "H5Z_FILTER_LZF" ","
	      : "id" ":" 32000
	      : "}"

.. productionlist::
   nbit_filter: "{"
              : "class" ":" "H5Z_FILTER_NBIT" ","
	      : "id" ":" 5
	      : "}"

.. productionlist::
   scaleoffset_filter: "{"
                     : "class" ":" "H5Z_FILTER_SCALEOFFSET" ","
		     : "id" ":" 6 ","
		     : "scaleType" ":" `scale_type` ","
		     : "scaleOffset" ":" `positive_integer`
		     : "}"
   scale_type:  "H5Z_SO_FLOAT_DSCALE"
             :| "H5Z_SO_FLOAT_ESCALE"
	     :| "H5Z_SO_INT"

.. productionlist::
   shuffle_filter: "{"
                 : "class" ":" "H5Z_FILTER_SHUFFLE" ","
		 : "id" ":" 2
		 : "}"

.. productionlist::
   szip_filter: "{"
              : "class" ":" "H5Z_FILTER_SZIP" ","
	      : "id" ":" 4 ","
	      : "bitsPerPixel" ":" `positive_integer` ","
	      : "coding" ":" `coding` ","
	      : "pixelsPerBlock" ":" `positive_integer` ","
	      : "pixelsPerScanline" ":" `positive_integer`
	      : "}"
   coding:  "H5_SZIP_EC_OPTION_MASK"
         :| "H5_SZIP_NN_OPTION_MASK"

.. productionlist::
   other_filter: "{"
               : "class" ":" "H5Z_FILTER_USER" ","
	       : "id" ":" `positive_integer` ","
	       : "parameters" ":" `positive_integer_array`
	       : "}"
