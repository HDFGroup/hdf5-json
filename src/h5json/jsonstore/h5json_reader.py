##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of H5Serv (HDF5 REST Server) Service, Libraries and      #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
import json
import logging

from ..objid import getCollectionForId, getUuidFromId

from ..hdf5dtype import createDataType
from ..array_util import jsonToArray
from .. import selections
from ..h5reader import H5Reader


class H5JsonReader(H5Reader):
    """
    This class can be used by HDF5DB to read content from an hdf5-json file
    """

    def __init__(
        self,
        filepath,
        app_logger=None
    ):
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

        super().__init__(filepath, app_logger=app_logger)

        with open(filepath) as f:
            text = f.read()

        # parse the json file
        h5json = json.loads(text)

        self._h5json = h5json

        if "root" not in h5json:
            raise Exception("no root key in input file")
        self._root_id = "g-" + h5json["root"]

    def close(self):
        pass

    def get_root_id(self):
        """ Return root id """
        return self._root_id

    def getObjectById(self, obj_id, include_attrs=True, include_links=True, include_values=False):
        """ return object with given id """
        collection = getCollectionForId(obj_id)
        if collection not in self._h5json:
            self.log.warning(f"getObjectById - collection: {collection} not found")
            return None
        json_objs = self._h5json[collection]
        obj_uuid = getUuidFromId(obj_id)
        if obj_uuid not in json_objs:
            self.log.warning(f"getObjectById - {obj_id} not found")
            return None
        json_obj = json_objs[obj_uuid]

        resp = {}
        # selectively copy from the db dict
        for k in json_obj:
            for k in ("shape", "type", "cpl", "dcpl"):
                if k in json_obj:
                    resp[k] = json_obj[k]
        if include_attrs and "attributes" in json_obj:
            attrs = {}
            attr_list = json_obj["attributes"]
            for item in attr_list:
                if "name" not in item:
                    self.log.warning(f"expected to find name key for {obj_id} attributes")
                    continue
                name = item["name"]
                attr = {}
                if "type" not in item:
                    raise KeyError(f"expected to find type key for attribute {name} of {obj_id}")
                attr["type"] = item["type"]
                if "shape" not in item:
                    raise KeyError(f"expected to find shape key for attribute {name} of {obj_id}")
                attr["shape"] = item["shape"]
                if "value" in item:
                    attr["value"] = item["value"]
                attrs[name] = attr
            resp["attributes"] = attrs

        if include_links and "links" in json_obj:
            links = {}
            link_list = json_obj["links"]
            for item in link_list:
                if "title" not in item:
                    self.log.warning(f"expected to find title key for {obj_id} links")
                    continue
                title = item["title"]
                link = {}
                for k in ("class", "file", "h5path"):
                    if k in item:
                        link[k] = item[k]
                if "collection" in item:
                    collection = item["collection"]
                    if "id" not in item:
                        self.log.warning(f"expected to find id key for {obj_id} link item")
                        continue
                    obj_uuid = item["id"]
                    if collection == "groups":
                        obj_id = "g-" + obj_uuid
                    elif collection == "datasets":
                        obj_id = "d-" + obj_uuid
                    elif collection == "datatypes":
                        obj_id = "t-" + obj_uuid
                    else:
                        self.log.warning(f"unexpected collection type: {collection}")
                        continue
                    item["id"] = obj_id
                links[title] = item
            resp["links"] = links

        if include_values and collection == "datasets" and "value" in json_obj:
            resp["value"] = json_obj["value"]

        return resp

    def getAttribute(self, obj_id, name, includeData=True):
        """
        Get attribute given an object id and name
        returns: JSON object
        """
        self.log.debug(f"getAttribute({obj_id}), [{name}], include_data={includeData})")
        json_obj = self.getObjectById(obj_id)
        if json_obj is None:
            return None
        if "attributes" not in json_obj:
            self.log.warning(f"obj: {obj_id} has no attributes collection")
            return None
        attributes = json_obj["attributes"]
        if name not in attributes:
            self.log.info(f"attr: [{name}] of {obj_id} not found")
            return None
        return attributes[name]

    def getDtype(self, obj_json):
        """ Return the dtype for the type given by obj_json """
        if "type" not in obj_json:
            raise KeyError("no type item found")
        type_item = obj_json["type"]
        if isinstance(type_item, str) and type_item.startswith("datatypes/"):
            # this is a reference to a committed type
            ctype_id = "t-" + getUuidFromId(type_item)
            ctype_json = self.getObjectById(ctype_id)
            if "type" not in ctype_json:
                raise KeyError(f"Unexpected datatype: {ctype_json}")
            # Use the ctype's item json
            type_item = ctype_json["type"]
        dtype = createDataType(type_item)
        return dtype

    def getDatasetValues(self, obj_id, sel=None, dtype=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """

        self.log.debug(f"getDatasetValues({obj_id}), sel={sel}")
        json_obj = self.getObjectById(obj_id, include_values=True)
        if json_obj is None:
            self.log.warning(f"no object found with id; {obj_id}")
            return None

        if "value" not in json_obj:
            self.log.warning(f"value key not found for {obj_id}")
            return None
        json_value = json_obj["value"]
        shape_json = json_obj["shape"]
        if shape_json["class"] == "H5S_NULL":
            self.log.warning("getDatasetValues called for null space object: {obj_id}")
            return None
        elif shape_json["class"] == "H5S_SCALAR":
            dims = ()
        else:
            dims = shape_json["dims"]

        arr = jsonToArray(dims, dtype, json_value)
        if sel is None or sel.select_type == selections.H5S_SELECT_ALL:
            pass  # just return the entire array
        elif isinstance(sel, selections.SimpleSelection):
            arr = arr[sel.slices]
        else:
            raise NotImplementedError("selection type not supported")

        return arr
