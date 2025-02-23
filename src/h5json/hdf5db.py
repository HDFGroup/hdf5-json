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
import time
import numpy as np
import logging
from .hdf5dtype import getTypeItem, createDataType, Reference, special_dtype
from .array_util import jsonToArray, bytesArrayToList
from .dset_util import make_new_dset, resize_dataset
from .objid import createObjId, getCollectionForId
from .apiversion import _apiver
from .reader.h5reader import H5Reader
from .writer.h5writer import H5Writer


class Hdf5db:
    """
    This class is used to manage id lookup tables for primary HDF objects (Groups, Datasets,
    and Datatypes).  By default all data is held in-memory.  Initialize with h5_reader to read from
    an HDF5 compatible storage pool, and or, h5_writer to write to an HDF5 compatible storage pool.
    """
     
    @staticmethod
    def getVersionInfo():
        versionInfo = {}
        versionInfo["hdf5-json-version"] = _apiver
        return versionInfo

    def __init__(
        self,
        h5_reader: H5Reader = None,
        h5_writer: H5Writer = None,
        app_logger = None,
    ):
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

        self._db = {}

        self._reader = h5_reader
        self._writer = h5_writer
    
        if self._reader:
            root_id = self._reader.get_root_id()
            group_json = self._reader.getObjectById(root_id)
        else:
            root_id = createObjId(obj_type="groups")
            # create a root group
            group_json = {"links": {}, "attributes": {}, "cpl": {}}
            group_json["created"] = time.time()

        if self._writer:
            self._writer.set_db(self)
        
        self._db[root_id] = group_json
        self._root_id = root_id

    def flush(self):
        """ write out any changes """
        if self._writer:
            self._writer.flush()
           
    def close(self):
        """ close reader and writer handles """
        self.log.info("Hdf5db __close")
        self.flush()
        if self._writer:                         
            self._writer.close()
        if self._reader:
            self._reader.close()
        self._root_id = None
        self._db = {}

    def __enter__(self):
        """ called on package init """
        self.log.info("Hdf5db __enter")
        return self

    def __exit__(self, type, value, traceback):
        """ called on package exit """
        self.log.info("Hdf5db __exit")
        self.close()
         

    def getObjectById(self, obj_id):
        """ return object with given id """
        if obj_id not in self._db:
            if self._reader:
                # load the obj from the reader
                obj_json = self._reader.getObjectById(obj_id)
                self._db[obj_id] = obj_json
            else:
                raise KeyError(f"obj_id: {obj_id} not found")
        obj_json = self._db[obj_id]

        return obj_json

    def getObjectIdByPath(self, h5path, parent_id=None):
        """ Return id for the given link path starting from parent_id if set,
        otherwise the root_id """

        if h5path == "/":
            return self._root_id  # just return root id

        if parent_id is None:
            parent_id = self._root_id
        self.log.debug(f"getObjectIdDByPath(h5path: {h5path} parent_id: {parent_id}")
        
        obj_json = self.getObjectById(parent_id)
        if obj_json is None:
            self.log.warning("getObjectIdDByPath - parent_id not found")
            raise KeyError("parent_id: {parent_id} not found")

        obj_id = parent_id
        searched_ids = set(obj_id)

        link_names = h5path.split('/')
        self.log.debug(f"link_names: {link_names}")
        for link_name in link_names:
            if not link_name:
                continue
            link_tgt = None
            self.log.debug(f"link_name: {link_name}")
            if not obj_id:
                break
            if 'links' not in obj_json:
                self.log.error(f"expected to find links key in: {obj_json}")
                raise KeyError(h5path)
            links = obj_json['links']
            self.log.debug(f"links: {links}")
            if link_name not in links:
                self.log.warning(f"link: {link_name} not found in {obj_id}")
                self.log.debug(f"links: {links}")
                raise KeyError(h5path)
            link_tgt = links[link_name]
            self.log.debug(f"link_tgt: {link_tgt}")
            link_class = link_tgt['class']
            obj_id = None
            obj_json = None
            if link_class == 'H5L_TYPE_HARD':
                # hard link
                obj_id = link_tgt['id']
                if obj_id in searched_ids:
                    self.log.warning(f"circular reference using path: {h5path}")
                    raise KeyError(h5path)
                obj_json = self.getObjectById(obj_id)
                searched_ids.add(obj_id)
            elif link_class == 'H5L_TYPE_SOFT':
                self.log.warning("getObjectIdByPath can't follow soft links")
            elif link_class == 'H5L_TYPE_EXTERNAL':
                self.log.warning("getObjectIdByPath can't follow external links")
            else:
                self.log.error(f"link type: {link_class} not supported")

            if not obj_id:
                self.log.warning(f"get_bypath {h5path} not found")
                raise KeyError(h5path)
        return obj_id
    
    def getObjectByPath(self, path):
        """ Get Object JSON at given path """
        obj_id = self.getObjectIDByPath(path)
        obj_json = self.getObjectById(obj_id)
        return obj_json    

    def getDtype(self, obj_id):
        """ Return numpy data type for given object id """
        if obj_id not in self._db:
            raise KeyError(f"{obj_id} not found")
        obj_json = self._db[obj_id]
        if "type" not in obj_json:
            # group id?
            raise TypeError(f"{obj_id} does not have a datatype")
        type_json = obj_json["type"]
        
        dtype = createDataType(type_json)
        return dtype
 
 
    def createCommittedType(self, datatype, cpl=None):
        """
        createCommittedType - creates new named datatype
        Returns item
        """
        self.log.info("createCommittedType")
        if cpl is None:
            cpl = {}
         
        ctype_id = createObjId(obj_type="datatypes", root_id=self._root_id)
        if isinstance(datatype, np.dtype):
            dt = datatype
        else:
            dt = createDataType(datatype)

        type_json = getTypeItem(dt)  # get canonical json description of datatype

        ctype_json = {"type": type_json, "attributes": {}, "cpl": cpl}
        ctype_json["created"] = time.time()
        ctype_json["modified"] = None
        self._db[ctype_id] = ctype_json
        return ctype_id
  

    def getAttribute(self, obj_id, name, includeData=True):
        """
        Get attribute given an object id and name
        returns: JSON object
        """

        obj_json = self.getObjectById(obj_id)
        attrs = obj_json["attributes"]
        
        if name not in attrs:
            msg = f"Attribute: [{name }] not found in object: {obj_id}"
            self.log.info(msg)
            return None
        
        attr_json = attrs[name]

        if includeData and "value" not in attr_json:
            # Reader may not have pre-loaded large attributes
            # fetch it now
            if not self._reader:
                raise RuntimeError(f"Expected to find value for attribute {name} of {obj_id}")
            attr_json = self._reader.get_attribute(obj_id, name)
            attr_json["value"] = attr_json  # this will update the _db
        
        return attr_json
    
    def getAttributes(self, obj_id):
        """
        Get attributes given an object id and name
        returns: JSON object
        """

        obj_json = self.getObjectById(obj_id)
        attrs = obj_json["attributes"]
         
        return attrs
    
    def getAttributeValue(self, obj_id, name):
        """ Return NDArray of the given attribute value """
        attr_json = self.getAttribute(obj_id, name)
        shape_json = attr_json["shape"]
        if shape_json["class"] == "H5S_NULL":
            # no value for empty shape attributes
            return None
        elif shape_json["class"] == "H5S_SCALAR":
            dims = ()
        else:
            dims = shape_json["dims"]
        dtype = createDataType(attr_json["type"])
        value = attr_json["value"]
        arr = jsonToArray(dims, dtype, value)
        return arr


    def createAttribute(self, obj_id, name, value, shape=None, dtype=None):
        """
        create an attribute - will override any existing attributes
        """
        
        # TBD: if dtype is a committed ref type, fetch it first
        # TBD: also, check special case for complex types

        if isinstance(dtype, str) and dtype.startswith("datatypes/"):
            ctype_id = dtype[len("datatypes/"):]
            if getCollectionForId(ctype_id) != "datatypes":
                raise TypeError(f"unexpected dtype value for createAttribute: {dtype}")
            if ctype_id not in self._db:
                raise KeyError(f"ctype: {ctype_id} not found")
            ctype_json = self.getObjectById(ctype_id)
            type_json = ctype_json["type"].copy()
            type_json["id"] = ctype_id
            dtype = createDataType(type_json)

        # First, make sure we have a NumPy array.   
        if isinstance(value, Reference) and dtype is None:
            dtype = special_dtype(ref=Reference)
        if shape == "H5S_NULL":
            if value:
                raise ValueError("Value can't be set for Null space attributes")
            if dtype is None:
                raise ValueError("Dtype must be set for Null space attributes")
            else:
                dtype = np.dtype(dtype)
        else:
            value = np.asarray(value, dtype=dtype, order='C')
            if dtype is None:
                dtype = value.dtype
            else:
                dtype = np.dtype(dtype)  # In case a string, e.g. 'i8' is passed
 
        # Where a top-level array type is requested, we have to do some
        # fiddling around to present the data as a smaller array of
        # sub-arrays.
        if value is not None:
            if dtype.subdtype is not None:
                subdtype, subshape = dtype.subdtype

                # Make sure the subshape matches the last N axes' sizes.
                if shape[-len(subshape):] != subshape:
                    raise ValueError(f"Array dtype shape {subshape} is incompatible with data shape {shape}")

                # New "advertised" shape and dtype
                shape = shape[0:len(shape) - len(subshape)]
                dtype = subdtype

            # Not an array type; make sure to check the number of elements
            # is compatible, and reshape if needed.
            else:
                if isinstance(shape, tuple):
                    if np.prod(shape) != np.prod(value.shape):
                        raise ValueError("Shape of new attribute conflicts with shape of data")

                    if shape != value.shape:
                        value = value.reshape(shape)

                # We need this to handle special string types.
                value = np.asarray(value, dtype=dtype)
            value_json = bytesArrayToList(value)
        else:
            value_json = None

        if shape is None:
            shape = value.shape
        if shape == "H5S_NULL":
            shape_json = {"class": "H5S_NULL"}
        elif len(shape) == 0:
            shape_json = {"class": "H5S_SCALAR"}
        else:
            shape_json = {"class": "H5S_SIMPLE"}
            shape_json["dims"] = list(shape)

        obj_json = self.getObjectById(obj_id)
        attrs_json = obj_json["attributes"]
        if name in attrs_json:
                # replace, update modified timestamp
            created = attrs_json["created"]
            modified = time.time()
        else:
            created = time.time()
            modified = None
        type_json = getTypeItem(dtype)
        # finally put it all together...
        attr_json = {"shape": shape_json, "type": type_json, "value": value_json}
        attr_json["created"] = created
        attr_json["modified"] = modified

        # slot into the obj_json["attrs"]
        attrs_json[name] = attr_json


    def deleteAttribute(self, obj_id, name):
        """ delete the given attribute """
        obj_json = self.getObjectById(obj_id)
        attrs_json = obj_json["attributes"]
        if name not in attrs_json:
            raise KeyError(f"attribute [{name}] not found in {obj_id}")
        del attrs_json[name]


    def getDatasetValues(self, obj_id, slices=Ellipsis, format="json"):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """
        self.log.info(f"getDatasetValues obj_id: {obj_id}, slices: {slices} format: {format}")
        #TBD
      

    def createDataset(
        self,
        shape=None,
        dtype=None,
        chunks=None,
        compression=None,
        shuffle=None,
        maxshape=None,
        compression_opts=None,
        fillvalue=None,
        cpl=None,
    ):
        """
        createDataset - creates new dataset given shape and datatype
        Returns obj_id
        """
        
        kwds = {}
        if chunks:
            kwds["chunks"] = chunks
        if compression:
            kwds["compression"] = compression
        if shuffle:
            kwds["shuffle"] = shuffle
        if compression_opts:
            kwds["compression_opts"] = compression_opts
        if maxshape:
            kwds["maxshape"] = maxshape
        if fillvalue:
            kwds["fillvalue"] = fillvalue
        if cpl:
            kwds["cpl"] = cpl
        dset_json = make_new_dset(shape=shape, dtype=dtype, **kwds)
 
        dset_id = createObjId("datasets", root_id=self._root_id)   
        self._db[dset_id] = dset_json 
        return dset_id


    def resizeDataset(self, dset_id, shape):
        """
        Resize existing Dataset
        """
        self.log.info(f"resizeDataset {dset_id}, {shape}")
        
        dset_json = self.getObjectById(dset_id)  # will throw exception if not found
        resize_dataset(dset_json, shape)
         

    def deleteObject(self, obj_id):
        """ Delete the given object """
        self.log.info(f"deleteObject: {obj_id}")
        if obj_id not in self._db:
            raise KeyError(f"Object {obj_id} not found for deletion")
        if obj_id == self._root_id:
            raise KeyError("Root group cannot be deleted")
        del self._db[obj_id]
        # TBD: add to pending deleted items
        
    def getLinks(self, grp_id):
        """ Get the links for the given group """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            raise KeyError(f"No links - {grp_id} not a group?")
        links = grp_json["links"]
        return links
      
    def getLink(self, grp_id, name):
        """ Get the given link """
        
        links = self.getLinks(grp_id)
        if name not in links:
            raise KeyError(f"Link [{name}] not found in {grp_id}")
        return links[name]
    
    def createHardLink(self, grp_id, name, tgt_id):
        """ Create a new hardlink """
        links = self.getLinks(grp_id)
        if name in links:
            self.deleteLink(grp_id, name)
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        link_json["created"] = time.time()
        links[name] = link_json

    def createSoftLink(self, grp_id, name, h5path):
        """ Create a soft link """
        links = self.getLinks(grp_id)
        if name in links:
            self.deleteLink(grp_id, name)
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": h5path}
        link_json["created"] = time.time()
        links[name] = link_json

    def createCustomLink(self, grp_id, name, link_json):
        """ create a custom link """
        links = self.getLinks(grp_id)
        if name in links:
            self.deleteLink(grp_id, name)
        if link_json.get("class") != "H5L_TYPE_USER_DEFINED":
            link_json["class"] = "H5L_TYPE_USER_DEFINED"
        link_json["created"] = time.time()
        links[name] = link_json


    def createExternalLink(self, grp_id, name, h5path, filepath):
        """ Create a external link link """
        links = self.getLinks(grp_id)
        if name in links:
            self.deleteLink(grp_id, name)
        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": h5path, "file": filepath}
        link_json["created"] = time.time()
        links[name] = link_json
 
    def deleteLink(self, grp_id, name):
        """ Delete the given link """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            raise KeyError(f"No links - {grp_id} not a group?")
        links = self.getLinks(grp_id)
        if name not in links:
            raise KeyError(f"Link [{name}] not found in {grp_id}")
        del links[name]
        grp_json["modified"] = time.time()
 

    def createGroup(self, cpl=None):
        """ Create a new group """

        grp_id = createObjId("groups", root_id=self._root_id)
        group_json = {"attributes": {}, "links": {}}
        if cpl:
            group_json["cpl"] = cpl
        else:
            group_json["cpl"] = {}
        group_json["created"] = time.time()
        group_json["modified"] = None
        self._db[grp_id] = group_json
        return grp_id
   

    def getCollection(self, col_type=None):
        obj_ids = []
        for obj_id in self._db:
            if not col_type or getCollectionForId(obj_id) == col_type:
                obj_ids.append(obj_id)
        return obj_ids

    def __len__(self):
        # return the number of objects
        return len(self._db)


    def __iter__(self):
        """ Iterate over object ids """

        for obj_id in self._db:
            yield obj_id


    def __contains__(self, obj_id):
        """ Test if a obj id  exists """
        return obj_id in self._db
