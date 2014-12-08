File
====

.. productionlist::
   file: "{"
       : "id" ":" `identifier` ","
       : "created" ":" `utc_datetime` ","
       : "lastModified" ":" `utc_datetime` ","
       : "root" ":" `id_reference` ","
       : "groups" ":" "{" `group_hashtable` "}" ","
       : "datasets" ":" "{" `dataset_hashtable` "}" ","
       : "datatypes" ":" "{" `datatype_hashtable` "}" ","
       : "userblock" ":" `byte_array` ","
       : "userblockSize" ":" `non_negative_integer` /power of 2, >= 512/ ","
       : "creationProperties" ":" `fcpl` ","
       : "driverInfo" ":" `file_driver_info` ","
       : "apiVersion" ":" "0.0.0"
       : "}"

.. productionlist::
   group_hashtable: `group` ("," `group`)*
   dataset_hashtable: `dataset` ("," `dataset`)*
   datatype_hashtable: `datatype_object` ("," `datatype_object`)*

.. productionlist::
   fcpl: "{"
       : "chunkStorageConfig" ":" `chunk_storage_config` ","
       : "freeListConfig" ":" `free_list_config` ","
       : "sizeConfig" ":" `size_config` ","
       : "sohmConfig" ":" `sohm_config` ","
       : "superBlockVersion" ":" `non_negative_integer` ","
       : "symbolTableConfig" ":" `sym_tab_config` ","
       : "}"

.. productionlist::
   chunk_storage_config: "{"
                       : "chunkStorageBTreeHalfRank" ":" `positive_integer`
 		       : "}"

.. productionlist::
   free_list_config: "{"
                   : "freeListVersion" ":" `non_negative_integer`
              	   : "}"

							
.. productionlist::
   size_config: "{"
              : "lengthSizeInBytes" ":" `positive_integer`
	      : "offsetSizeInBytes" ":" `positive_integer`
	      : "}"

.. productionlist::
   sohm_config: "{"
              : "maxList" ":" `non_negative_integer` ","
              : "minBTree" ":" `non_negative_integer` ","
              : "version" ":" `non_negative_integer` ","
	      : "indexes" ":" "[" `sohm_index_list` "]"
              : "}"
   sohm_index_list: `sohm_index` ("," `sohm_index`)*
   sohm_index: "{"
             : "messageTypes" ":" "[" `sohm_message_type_list` "]" ","
	     : "minMessageSize" ":" `non_negative_integer`
             : "}"
   sohm_message_type_list: `sohm_message_type` ("," `sohm_message_type`)*
   sohm_message_type:  "H5O_SHMESG_ATTR_FLAG"
                    :| "H5O_SHMESG_DTYPE_FLAG"
		    :| "H5O_SHMESG_FILL_FLAG"
		    :| "H5O_SHMESG_PLINE_FLAG"
		    :| "H5O_SHMESG_SDSPACE_FLAG"
	 
.. productionlist::
   sym_tab_config: "{"
                 : "nodeSize" ":" `positive_integer`
		 : "treeRank" ":" `positive_integer`
		 : "version" ":" `non_negative_integer`
		 : "}"

.. productionlist::
   file_driver_info: `family_driver_info` | `multi_driver_info`
   family_driver_info: "{"
                     : "memberSize" ":" `positive_integer`
                     : "}"
   multi_driver_info: "[" `data_distribution_list` "]"
   data_distribution_list: `data_item` ("," `data_item`)*
   data_item: "{"
            : "dataMap" ":" `data_kind` ","
	    : "fileName" ":" `unicode_string` ","
	    : "address"  ":" `positive_integer` ","
	    : "relaxFlag" ":" false | true 
            : "}"
   data_kind:  "H5FD_MEM_SUPER"
            :| "H5FD_MEM_BTREE"
	    :| "H5FD_MEM_DRAW"
	    :| "H5FD_MEM_GHEAP"
	    :| "H5FD_MEM_LHEAP"
	    :| "H5FD_MEM_OHDR"
