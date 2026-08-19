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
import os
from os import stat as os_stat
import time

from ..objid import getUuidFromId, getCollectionForId, createObjId
from ..hdf5dtype import createDataType, getItemSize
from ..array_util import bytesArrayToList, jsonToArray
from .. import selections
from ..storage_plugin import StoragePlugin


class H5JsonPlugin(StoragePlugin):
    """
    This class reads from and writes to an h5json (.json) file.  A single in-memory `self.json`
    dict is the source of truth for both reading and writing - it starts either loaded from an
    existing file (append=True and the file already exists) or empty (a fresh file), and is
    progressively updated by flush()'s dump methods, so a read always sees whatever this same
    instance has most recently flushed.
    """

    def __init__(
        self,
        filepath,
        append=False,
        data_limit=None,
        indent=4,
        read_only=False,
        app_logger=None
    ):
        no_data = True if data_limit == 0 else False
        super().__init__(filepath, append=append, no_data=no_data, read_only=read_only, app_logger=app_logger)
        self.json = {}
        self._data_limit = data_limit
        self._root_id = None
        self._root_uuid = None
        self._indent = indent
        # True until the first flush() completes - forces every group/dataset/datatype to be
        # (re-)dumped on that first flush, even ones not marked dirty. False for append mode,
        # matching H5pyPlugin: objects loaded from an existing file keep their loaded entry
        # until something actually makes them dirty.
        self._init = False if (append or read_only) else True

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """ open the json file - loading its content if appending to an existing file,
        otherwise starting from an empty store """
        if self._root_id:
            return self._root_id  # already open

        if (self._append or self._read_only) and self._filepath and os.path.isfile(self._filepath):
            with open(self._filepath) as f:
                text = f.read()
            self.json = json.loads(text)
            if "root" not in self.json:
                raise Exception("no root key in input file")
            self._root_id = "g-" + self.json["root"]
            if self.db.root_id and self.db.root_id != self._root_id:
                self.log.warning("h5json root id doesn't match db root id")
                raise IOError("root id mismatch")
        else:
            self.json = {}
            if self.db.root_id:
                self._root_id = self.db.root_id
            else:
                self._root_id = createObjId(obj_type="groups")

        return self._root_id

    def close(self):
        """ close storage handle.

        Doesn't flush - Hdf5db.close() (the only caller) always calls
        Hdf5db.flush() immediately beforehand, which itself calls this
        plugin's flush(); re-flushing here would be redundant (and, for a
        stdout-destined store, would print the dump a second time). """
        self._root_id = None
        self.json = {}

    def isClosed(self):
        """ return closed status """
        return False if self._root_id else True

    def get_root_id(self):
        """ Return root id """
        return self._root_id

    # ------------------------------------------------------------------
    # read-side retrieval - reads from self.json, which always reflects
    # whatever this instance last loaded or flushed
    # ------------------------------------------------------------------

    def getObjectById(self, obj_id, include_attrs=True, include_links=True, include_values=False):
        """ return object with given id """
        collection = getCollectionForId(obj_id)
        if collection not in self.json:
            self.log.warning(f"getObjectById - collection: {collection} not found")
            return None
        json_objs = self.json[collection]
        obj_uuid = getUuidFromId(obj_id)
        if obj_uuid not in json_objs:
            self.log.warning(f"getObjectById - {obj_id} not found")
            return None
        json_obj = json_objs[obj_uuid]

        resp = {}
        # selectively copy from the db dict
        for k in json_obj:
            for k in ("shape", "type", "cpl", "dcpl", "creationProperties", "encoding"):
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
                    if "encoding" in item:
                        attr["encoding"] = item["encoding"]
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
                    link_collection = item["collection"]
                    if "id" not in item:
                        self.log.warning(f"expected to find id key for {obj_id} link item")
                        continue
                    tgt_uuid = item["id"]
                    if link_collection == "groups":
                        link_id = "g-" + tgt_uuid
                    elif link_collection == "datasets":
                        link_id = "d-" + tgt_uuid
                    elif link_collection == "datatypes":
                        link_id = "t-" + tgt_uuid
                    else:
                        self.log.warning(f"unexpected collection type: {link_collection}")
                        continue
                    link["id"] = link_id
                links[title] = link
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
        attr_json = attributes[name]
        if not includeData and ("value" in attr_json or "encoding" in attr_json):
            attr_json = dict(attr_json)
            attr_json.pop("value", None)
            attr_json.pop("encoding", None)
        return attr_json

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

    def getDatasetValues(self, obj_id, sel=None, dtype=None, query=None):
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

        if query is not None:
            # json store doesn't support query
            raise NotImplementedError("getDatasetValues with query not implemented for H5JsonPlugin")

        arr = jsonToArray(dims, dtype, json_value)

        if sel is None or sel.select_type == selections.H5S_SEL_ALL:
            pass  # just return the entire array
        elif isinstance(sel, selections.SimpleSelection):
            arr = arr[sel.slices]
        else:
            raise NotImplementedError("selection type not supported")

        return arr

    # ------------------------------------------------------------------
    # write-side dump methods - populate self.json from the current
    # Hdf5db state
    # ------------------------------------------------------------------

    def getAliasList(self, obj_id):
        """ return list of alias """

        return self.db.getPathsForObjectId(obj_id)

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
            if "encoding" in item:
                response["encoding"] = item["encoding"]
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
            response["creationProperties"] = item["cpl"]
        elif "creationProperties" in item:
            response["creationProperties"] = item["creationProperties"]
        attributes = self.dumpAttributes(obj_id)
        if attributes:
            response["attributes"] = attributes
        links = self.dumpLinks(obj_id)
        if links:
            response["links"] = links
        return response

    def dumpGroups(self):
        groups = self.json.setdefault("groups", {})

        root_uuid = getUuidFromId(self._root_uuid)
        if self._init or root_uuid not in groups or self.db.is_dirty(self._root_uuid):
            groups[root_uuid] = self.dumpGroup(self._root_uuid)

        obj_ids = self.db.getCollection("groups")
        live_uuids = {root_uuid}
        for obj_id in obj_ids:
            if obj_id == self._root_uuid:
                continue
            obj_uuid = getUuidFromId(obj_id)
            live_uuids.add(obj_uuid)
            if self._init or obj_uuid not in groups or self.db.is_dirty(obj_id):
                groups[obj_uuid] = self.dumpGroup(obj_id)

        # drop entries for groups deleted since the last flush
        for obj_uuid in list(groups):
            if obj_uuid not in live_uuids:
                del groups[obj_uuid]

    def dumpDataset(self, obj_id):
        response = {}
        self.log.info("dumpDataset: " + obj_id)
        item = self.db.getObjectById(obj_id)
        alias = self.getAliasList(obj_id)
        response["alias"] = alias

        type_item = item["type"]
        response["type"] = type_item
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
        if self._data_limit is not None:
            item_size = getItemSize(type_item)
            if item_size == "H5T_VARIABLE":
                item_size = 1024  # assume average size for variable length types
            total_size = item_size * num_elements

            if total_size > self._data_limit:
                self.log.info(f"skipping data dump for dataset {obj_id} with {num_elements} elements")
        if self._data_limit is None or total_size <= self._data_limit:
            if num_elements > 0:
                sel_all = selections.select(dims, ...)
                arr = self.db.getDatasetValues(obj_id, sel_all)
                response["value"] = bytesArrayToList(arr)  # dump values unless header flag was passed
                if type_item.get("class") == "H5T_OPAQUE":
                    response["encoding"] = "base64"
        return response

    def dumpDatasets(self):
        obj_ids = self.db.getCollection("datasets")
        datasets = self.json.setdefault("datasets", {})

        live_uuids = set()
        for obj_id in obj_ids:
            obj_uuid = getUuidFromId(obj_id)
            live_uuids.add(obj_uuid)
            if self._init or obj_uuid not in datasets or self.db.is_dirty(obj_id):
                datasets[obj_uuid] = self.dumpDataset(obj_id)

        # drop entries for datasets deleted since the last flush
        for obj_uuid in list(datasets):
            if obj_uuid not in live_uuids:
                del datasets[obj_uuid]

        if not datasets:
            del self.json["datasets"]

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
        datatypes = self.json.setdefault("datatypes", {})

        live_uuids = set()
        for obj_id in obj_ids:
            obj_uuid = getUuidFromId(obj_id)
            live_uuids.add(obj_uuid)
            if self._init or obj_uuid not in datatypes or self.db.is_dirty(obj_id):
                datatypes[obj_uuid] = self.dumpDatatype(obj_id)

        # drop entries for datatypes deleted since the last flush
        for obj_uuid in list(datatypes):
            if obj_uuid not in live_uuids:
                del datatypes[obj_uuid]

        if not datatypes:
            del self.json["datatypes"]

    def dumpFile(self):
        self._root_uuid = self.db.getObjectIdByPath("/")

        db_version_info = self.db.getVersionInfo()

        self.json["apiVersion"] = db_version_info["hdf5-json-version"]
        self.json["root"] = getUuidFromId(self._root_uuid)

        self.dumpGroups()

        self.dumpDatasets()

        self.dumpDatatypes()
        indent = self._indent
        ensure_ascii = True
        if self._filepath:
            with open(self._filepath, 'w', encoding='utf-8') as f:
                json.dump(self.json, f, ensure_ascii=ensure_ascii, indent=indent)
        else:
            print(json.dumps(self.json, sort_keys=True, ensure_ascii=ensure_ascii, indent=indent))
        self._lastModified = time.time()  # update timestamp

    def flush(self):
        """ Write dirty items.

        flush() may be called more than once per session (e.g. Hdf5db's
        periodic auto-flush, or close() flushing before its own final
        flush()). dumpGroups()/dumpDatasets()/dumpDatatypes() only recompute
        entries for objects that are new/dirty/resized (or deleted) since
        the previous flush - see the "_init" handling there - since
        recomputing an unchanged dataset's value via Hdf5db.getDatasetValues()
        would incorrectly return a zero/fill-value array once its pending
        update has already been cleared by an earlier flush (there being no
        way to re-fetch the previously-written value otherwise). """

        if not self._root_id:
            msg = "flush called prior to open"
            self.log.warning(msg)
            raise IOError(msg)
        if self._read_only:
            if self.db.new_objects or self.db.dirty_objects:
                # a read_only plugin must never write to storage, but in-memory-only
                # edits made against it (e.g. transient annotations the caller never
                # intends to persist) are fine to just leave un-flushed
                self.log.warning("read_only plugin: not persisting pending in-memory changes")
                return False
            return True  # nothing to persist, and never anything to initialize

        self.log.info("flush")
        self.dumpFile()
        self._init = False

        return True

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

    def getFilters(self, compressors_only=False):
        """ return empty list of filters """

        return ()
