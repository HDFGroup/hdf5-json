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
from os import stat as os_stat
import time

from ..h5writer import H5Writer
from ..objid import getUuidFromId, getCollectionForId, createObjId
from ..array_util import bytesArrayToList
from .. import selections


class H5JsonWriter(H5Writer):
    """
    This abstract class defines properties and methods that the Hdf5db class uses for writing to an HDF5
    compatible storage medium.
    """

    def __init__(
        self,
        filepath,
        append=False,
        no_data=False,
        app_logger=None
    ):
        super().__init__(filepath, append=append, no_data=no_data, app_logger=app_logger)
        if append:
            raise ValueError("H5JsonWriter does not support append mode")
        self.alias_db = {}
        self.json = {}
        self._root_id = None

    def flush(self):
        """ Write dirty items """

        if not self._root_id:
            msg = "flush called prior to open"
            self.log.warning(msg)
            raise IOError(msg)

        self.log.info("flush")
        self.dumpFile()
        return True

    def open(self):
        """ file open """
        # no incremental updates with h5json writer, so just fetch the root_id here
        if self.db.root_id:
            self._root_id = self.db.root_id
        else:
            self._root_id = createObjId(obj_type="groups")
        return self._root_id

    def close(self):
        """ close storage handle """
        self.flush()
        self._root_id = None

    def isClosed(self):
        """ return closed status """
        return False if self._root_id else True

    def getAliasList(self, obj_id):
        """ return list of alias """
        if obj_id not in self.alias_db:
            self.alias_db[obj_id] = []
        return self.alias_db[obj_id]

    def updateAliasList(self):
        """ update the alias list for each object """
        # clear exiting aliases
        obj_ids = self.db.getCollection()
        for obj_id in obj_ids:
            self.alias_db[obj_id] = []

        self._setAlias(self._root_uuid, set(), "/")

    def _setAlias(self, obj_id, id_set, h5path):
        """ add the given h5path to the object's alias list
            If the object is a group, recurse through each hard link """
        obj_json = self.db.getObjectById(obj_id)
        alias_list = self.getAliasList(obj_id)
        if h5path in alias_list:
            return  # nothing to do
        alias_list.append(h5path)
        if getCollectionForId(obj_id) != "groups":
            return  # done
        id_set.add(obj_id)  # keep track of objects we've visited to avoid loops
        links = obj_json["links"]
        if h5path[-1] != '/':
            h5path += '/'

        for link_name in links:
            link_json = links[link_name]
            if link_json["class"] == "H5L_TYPE_HARD":
                tgt_id = link_json["id"]
                if tgt_id in id_set:
                    self.log.info("_setAlias - circular loop found")
                else:
                    self._setAlias(tgt_id, id_set, f"{h5path}{link_name}")
        id_set.remove(obj_id)

    def dumpAttribute(self, obj_id, attr_name):
        self.log.info(f"dumpAttribute: [{attr_name}]")
        item = self.db.getAttribute(obj_id, attr_name)
        response = {"name": attr_name}
        response["type"] = item["type"]
        response["shape"] = item["shape"]

        if "value" not in item:
            self.log.warning(f"no value key in attribute: {attr_name}")
        else:
            # dump values unless header -D was passed
            response["value"] = item["value"]
        return response

    def dumpAttributes(self, obj_id):
        attrs = self.db.getAttributes(obj_id)
        self.log.info(f"dumpAttributes: {obj_id}")
        items = []
        for attr_name in attrs:
            item = self.dumpAttribute(obj_id, attr_name)
            items.append(item)

        return items

    def dumpLink(self, obj_id, name):
        item = self.db.getLink(obj_id, name)
        response = {"class": item["class"]}
        if "id" in item:
            tgt_id = item["id"]
            response["collection"] = getCollectionForId(tgt_id)
            response["id"] = getUuidFromId(tgt_id)

        for key in item:
            if key in ("id", "created", "modified"):
                continue
            response[key] = item[key]
        response["title"] = name
        return response

    def dumpLinks(self, obj_id):
        links = self.db.getLinks(obj_id)
        items = []
        for link_name in links:
            item = self.dumpLink(obj_id, link_name)
            items.append(item)
        return items

    def dumpGroup(self, obj_id):
        item = self.db.getObjectById(obj_id)
        response = {}

        alias = self.getAliasList(obj_id)
        response["alias"] = alias

        if "cpl" in item:
            item["creationProperties"] = item["cpl"]
        attributes = self.dumpAttributes(obj_id)
        if attributes:
            response["attributes"] = attributes
        links = self.dumpLinks(obj_id)
        if links:
            response["links"] = links
        return response

    def dumpGroups(self):
        groups = {}
        item = self.dumpGroup(self._root_uuid)
        root_uuid = getUuidFromId(self._root_uuid)
        groups[root_uuid] = item
        obj_ids = self.db.getCollection("groups")
        for obj_id in obj_ids:
            if obj_id == self._root_uuid:
                continue
            item = self.dumpGroup(obj_id)
            obj_uuid = getUuidFromId(obj_id)
            groups[obj_uuid] = item

        self.json["groups"] = groups

    def dumpDataset(self, obj_id):
        response = {}
        self.log.info("dumpDataset: " + obj_id)
        item = self.db.getObjectById(obj_id)
        alias = self.getAliasList(obj_id)
        response["alias"] = alias

        response["type"] = item["type"]
        shapeItem = item["shape"]
        shape_rsp = {}
        num_elements = 1
        shape_rsp["class"] = shapeItem["class"]
        if shapeItem["class"] == "H5S_NULL":
            dims = None
            num_elements = 0
        elif shapeItem["class"] == "H5S_SCALAR":
            dims = ()
            num_elements = 1
        else:
            shape_rsp["dims"] = shapeItem["dims"]
            dims = tuple(shapeItem["dims"])
            for extent in dims:
                num_elements *= extent

        if "maxdims" in shapeItem:
            maxdims = []
            for dim in shapeItem["maxdims"]:
                if dim == 0:
                    maxdims.append("H5S_UNLIMITED")
                else:
                    maxdims.append(dim)
            shape_rsp["maxdims"] = maxdims
        response["shape"] = shape_rsp

        if "cpl" in item:
            response["creationProperties"] = item["cpl"]

        attributes = self.dumpAttributes(obj_id)
        if attributes:
            response["attributes"] = attributes

        if not self.no_data:
            if num_elements > 0:
                sel_all = selections.select(dims, ...)
                arr = self.db.getDatasetValues(obj_id, sel_all)
                response["value"] = bytesArrayToList(arr)  # dump values unless header flag was passed
        return response

    def dumpDatasets(self):
        obj_ids = self.db.getCollection("datasets")
        if obj_ids:
            datasets = {}
            for obj_id in obj_ids:
                item = self.dumpDataset(obj_id)
                obj_uuid = getUuidFromId(obj_id)
                datasets[obj_uuid] = item

            self.json["datasets"] = datasets

    def dumpDatatype(self, obj_id):
        response = {}
        item = self.db.getObjectById(obj_id)
        alias = self.getAliasList(obj_id)
        response["alias"] = alias
        response["type"] = item["type"]
        if "cpl" in item:
            response["creationProperties"] = item["cpl"]
        attributes = self.dumpAttributes(obj_id)
        if attributes:
            response["attributes"] = attributes
        return response

    def dumpDatatypes(self):
        obj_ids = self.db.getCollection("datatypes")
        if obj_ids:
            datatypes = {}
            for obj_id in obj_ids:
                item = self.dumpDatatype(obj_id)
                obj_uuid = getUuidFromId(obj_id)
                datatypes[obj_uuid] = item

            self.json["datatypes"] = datatypes

    def dumpFile(self):
        self._root_uuid = self.db.getObjectIdByPath("/")

        db_version_info = self.db.getVersionInfo()

        self.json["apiVersion"] = db_version_info["hdf5-json-version"]
        self.json["root"] = getUuidFromId(self._root_uuid)

        self.updateAliasList()  # create alias_db with obj_id to alias list dict

        self.dumpGroups()

        self.dumpDatasets()

        self.dumpDatatypes()
        indent = 4
        ensure_ascii = True
        if self._filepath:
            with open('data.json', 'w', encoding='utf-8') as f:
                json.dump(self.json, f, ensure_ascii=ensure_ascii, indent=indent)
        else:
            print(json.dumps(self.json, sort_keys=True, ensure_ascii=ensure_ascii, indent=indent))
        self._lastModified = time.time()  # update timestamp

    def getStats(self):
        """ return a dictionary object with at minimum the following keys:
            'created': creation time
            'lastModified': modificationTime
            'owner': owner name
        """
        stat_info = os_stat(self.filepath)
        stats = {}
        stats['created'] = stat_info.st_ctime
        stats["lastModified"] = stat_info.st_mtime
        stats['owner'] = stat_info.st_uid  # TBD: convert to username?
        return stats
