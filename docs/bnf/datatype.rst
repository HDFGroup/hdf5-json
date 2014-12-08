.. rubric:: Datatype

.. productionlist::
   datatype_object: `identifier` ":" "{"
		  : "alias" ":" `hdf5_path_name`
		  :| `hdf5_path_name_list` ","
                  : "type" ":" `datatype`
                  : "}"
   datatype: `array_datatype`
            :| `bitfield_datatype`
            :| `compound_datatype`
            :| `enumeration_datatype`
            :| `floating_point_datatype`
            :| `integer_datatype`
            :| `opaque_datatype`
            :| `reference_datatype`
            :| `string_datatype`
	    :| `vlen_datatype`


.. productionlist::
   array_datatype: "{"
                 : "class" ":" "H5T_ARRAY" ","
                 : "base" ":" `datatype` ","
		 : "dims" ":" `dims_array`
		 : "}"


.. productionlist::
   bitfield_datatype: "{"
                    : "class" ":" "H5T_BITFIELD" ","
                    : ( `bitfield_predef` | `bitfield_user` )
                    : "}"
   bitfield_predef: "base" ":"
                  : ( "H5T_STD_B8BE"  | "H5T_STD_B8LE"
                  :|  "H5T_STD_B16BE" | "H5T_STD_B16LE"
		  :|  "H5T_STD_B32BE" | "H5T_STD_B32LE"
		  :|  "H5T_STD_B64BE" | "H5T_STD_B64LE" )
   bitfield_user : "bitOffset" ":" `non_negative_integer` ","
		 : "byteOrder" ":" `byte_order` ","
		 : "lsbPad" ":" `bit_padding` ","
		 : "msbPad" ":" `bit_padding` ","
		 : "precision" ":" `positive_integer` ","
		 : "size" ":" `positive_integer`
   bit_padding : "H5T_PAD_ZERO"
               :| "H5T_PAD_ONE"
	       :| "H5T_PAD_BACKGROUND"
   byte_order : "H5T_ORDER_LE" | "H5T_ORDER_BE"


.. productionlist::
   compound_datatype: "{"
                    : "class" ":" "H5T_COMPOUND" ","
	   	    : "fields" ":" "[" `field_list` "]"
		    : "}"
   field_list: `field_def` ("," `field_def`)*
   field_def: "{"
            : "name" ":" `ascii_string` ","
	    : "type" ":" `datatype`
	    : "}"


.. productionlist::
   enumeration_datatype: "{"
		       : "base" ":" `integer_datatype` ","
                       : "class" ":" "H5T_ENUM" ","
		       : "members" ":" "[" `enum_member_list` "]"
		       : "}"
   enum_member_list: `enum_member` ("," `enum_member`)*
   enum_member: "{" `ascii_string` ":" `integer` "}"


.. productionlist::
   floating_point_datatype: "{"
                          : "class" ":" "H5T_FLOAT" ","
                          : ( `float_predef` | `float_user` )
                          : "}"
   float_predef: "base" ":"
               : ( "H5T_IEEE_F32BE" | "H5T_IEEE_F32LE"
               :|  "H5T_IEEE_F64BE" | "H5T_IEEE_F64LE" )
   float_user: "{"
             : "bitOffset" ":" `non_negative_integer` ","
	     : "byteOrder" ":" `byte_order` ","
	     : "expBias" ":" `positive_integer` ","
	     : "expBits" ":" `positive_integer` ","
	     : "expBitPos" ":" `positive_integer` ","
	     : "intlbPad" ":" `bit_padding` ","
	     : "lsbPad" ":" `bit_padding` ","
	     : "mantBits" ":" `positive_integer` ","
	     : "mantBitPos" ":" `non_negative_integer` ","
	     : "mantNorm" ":" `mant_norm` ","
	     : "msbitPad" ":" `bit_padding` ","
	     : "precision" ":" `positive_integer` ","
	     : "signBitPos" ":" `positive_integer` ","
	     : "size" ":" `positive_integer`
	     : "}"
   mant_norm: "H5T_NORM_IMPLIED"
            :| "H5T_NORM_MSBSET"
	    :| "H5T_NORM_NONE"

.. productionlist::
   integer_datatype: "{"
                   : "class" ":" "H5T_INTEGER" ","
                   : ( `integer_predef` | `integer_user` )
                   : "}"
   integer_predef: "base" ":"
                 : ( "H5T_STD_I8BE"  | "H5T_STD_I8LE"
                 :|  "H5T_STD_I16BE" | "H5T_STD_I16LE"
		 :|  "H5T_STD_I32BE" | "H5T_STD_I32LE"
		 :|  "H5T_STD_I64BE" | "H5T_STD_I64LE"
		 :|  "H5T_STD_U8BE"  | "H5T_STD_U8LE"
		 :|  "H5T_STD_U16BE" | "H5T_STD_U16LE"
		 :|  "H5T_STD_U32BE" | "H5T_STD_U32LE"
		 :|  "H5T_STD_U64BE" | "H5T_STD_U64LE" )
   integer_user: "{"
	       : "bitOffset" ":" `non_negative_integer` ","
	       : "byteOrder" ":" `byte_order` ","
	       : "lsbPad" ":" `bit_padding` ","
	       : "msbPad" ":" `bit_padding` ","
	       : "precision" ":" `positive_integer` ","
	       : "signType" ":" `sign_type` ","
	       : "size" ":" `positive_integer`
	       : "}"
   sign_type: "H5T_SGN_NONE" | "H5T_SGN_2"

.. productionlist::
   opaque_datatype: "{"
                  : "class" ":" "H5T_OPAQUE" ","
                  : "size" ":" `positive_integer` ","
		  : "tag"  ":" `ascii_string`
                  : "}"


.. productionlist::
   reference_datatype: "{"
                     : "class" ":" "H5T_REFERENCE" ","
                     : "base" ":"
                     : ( "H5T_STD_REF_OBJ"
                     :|  "H5T_STD_REF_DSETREG" )
   object_reference_value: `dataset_ref`
                         :| `datatype_object_ref`
			 :| `group_ref`
   region_reference_value: "{"
                         : "dataset" ":" `dataset_ref` ";"
			 : "selection" ":" `dataspace_selection`
                         : "}"
   dataset_ref: `url_path` /\/datasets/`id_reference`/
   datatype_object_ref: `url_path` /\/datatypes/`id_reference`/
   group_ref: `url_path` /\/groups/`id_reference`/
					 
.. productionlist::
   string_datatype: "{"
                  : "charSet" ":" `char_encoding`
                  : "class" ":" "H5T_STRING" ","
		  : "length" " ":" `string_length`
		  : "strPad" ":" `string_padding` ","
                  : "}"
   char_encoding: "H5T_CSET_ASCII" | "H5T_CSET_UTF8"
   string_length: `positive_integer` | "H5T_VARIABLE"
   string_padding: "H5T_STR_NULLTERM"
                 :| "H5T_STR_NULLPAD"
		 :| "H5T_STR_SPACEPAD"


.. productionlist::
   vlen_datatype: "{"
                : "class" ":" "H5T_VLEN" ","
		: "base" ":" `datatype`
		: "}"
