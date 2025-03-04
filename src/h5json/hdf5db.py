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
from .dset_util import resize_dataset
from .objid import createObjId, getCollectionForId
from . import selections
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
        
        self._new_objects = set()  # set of obj_id's
        self._dirty_objects = set()  # set of obj_id's
    
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

    @property
    def db(self):
        """ return object db dictionary """
        return self._db
    
    @property
    def reader(self):
        """ return reader instance """
        return self._reader
    
    @property
    def writer(self):
        """ return writer instance """
        return self._writer
    
    @property
    def root_id(self):
        """ return root uuid """
        return self._root_id
    
    def is_new(self, obj_id):
        """ return true if this is a new object (has not been persisted) """
        return obj_id in self._new_objects
    
    def is_dirty(self, obj_id):
        """ return true if this object has been modified """
        if self.is_new(obj_id):
            return True
        return obj_id in self._dirty_objects
    
    def make_dirty(self, obj_id):
        """ Mark the object as dirty and update the lastModified timestamp """
        if self.is_new(obj_id):
            # object hasn't been initially written yet, just return
            return
        if obj_id not in self.db:
            self.log.error("make dirty called on deleted object")
            raise KeyError(f"obj_id: {obj_id} not found")
        if self.db[obj_id] is None:
            # object deleted, just return
            return
        obj_json = self.db[obj_id]
        obj_json["lastModified"] = time.time()
        self._dirty_objects.add(obj_id)


    def flush(self):
        """ write out any changes """
        if not self.writer:
            return  # nothing to do
        
        print("self._new_objects:", self._new_objects)
        print("self._dirty_objects:", self._dirty_objects)
        obj_ids = self._new_objects.union(self._dirty_objects)
        print(f"hdf5db_flush {len(obj_ids)} objects")

        if not self.writer.flush():
            # flush not successful, don't clear dirty set
            return  


        for obj_id in obj_ids:
            obj_json = self._db[obj_id]
            if "values" in obj_json:
                obj_json["values"] = []

        # reset new and dirty sets
        self._new_objects = set()
        self._dirty_objects = set()
           
    def close(self):
        """ close reader and writer handles """
        self.log.info("Hdf5db __close")
        self.flush()
        if self.writer:                         
            self.writer.close()
        if self.reader:
            self.reader.close()
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
        if obj_id not in self.db:
            if self.reader:
                # load the obj from the reader
                obj_json = self.reader.getObjectById(obj_id)
                self.db[obj_id] = obj_json
            else:
                raise KeyError(f"obj_id: {obj_id} not found")
        obj_json = self.db[obj_id]

        return obj_json

    def getObjectIdByPath(self, h5path, parent_id=None):
        """ Return id for the given link path starting from parent_id if set,
        otherwise the root_id """

        if h5path == "/":
            return self.root_id  # just return root id

        if parent_id is None:
            parent_id = self.root_id
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
        obj_id = self.getObjectIdByPath(path)
        obj_json = self.getObjectById(obj_id)
        return obj_json    

    def getDtype(self, obj_id):
        """ Return numpy data type for given object id """
        if obj_id not in self.db:
            raise KeyError(f"{obj_id} not found")
        obj_json = self.db[obj_id]
        if "type" not in obj_json:
            # group id?
            raise TypeError(f"{obj_id} does not have a datatype")
        type_json = obj_json["type"]
        
        dtype = createDataType(type_json)
        return dtype
 
 
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
        if attrs[name] == None:
            msg = f"Attribute: [{name}] has been deleted"
            self.log.info(None)
            return None
        
        attr_json = attrs[name]

        if includeData and "value" not in attr_json:
            # Reader may not have pre-loaded large attributes
            # fetch it now
            if not self.reader:
                raise RuntimeError(f"Expected to find value for attribute {name} of {obj_id}")
            attr_json = self.reader.get_attribute(obj_id, name)
            attr_json["value"] = attr_json  # this will update the _db
        
        return attr_json
    
    def getAttributes(self, obj_id):
        """
        Get attributes given an object id and name
        returns: JSON object
        """

        obj_json = self.getObjectById(obj_id)
        attrs = obj_json["attributes"]
        names = []
        for name in attrs:
            if attrs[name] != None:
                names.append(name)
         
        return names
    
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
            if ctype_id not in self.db:
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
            # replace, keep, created timestamp
            created = attrs_json["created"]
        else:
            created = time.time()
        type_json = getTypeItem(dtype)
        # finally put it all together...
        attr_json = {"shape": shape_json, "type": type_json, "value": value_json}
        attr_json["created"] = created

        # slot into the obj_json["attrs"]
        attrs_json[name] = attr_json

        # mark object as dirty
        self.make_dirty(obj_id)


    def deleteAttribute(self, obj_id, name):
        """ delete the given attribute """
        obj_json = self.getObjectById(obj_id)
        attrs_json = obj_json["attributes"]
        if name not in attrs_json:
            raise KeyError(f"attribute [{name}] not found in {obj_id}")
        attrs_json[name] = None  # mark key for deletion
        
        self.make_dirty(obj_id)


    def getDatasetValues(self, dset_id, sel):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """
        self.log.info(f"getDatasetValues dset_id: {dset_id}, sel: {sel}")
        dset_json = self.getObjectById(dset_id)
        shape_json = dset_json["shape"]
        if not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")
       
        if shape_json["class"] == "H5S_NULL":
            return None

        if shape_json["class"] == "H5S_SCALAR":
            if sel.select_type != sel.H5S_SELECT_ALL:
                # TBD: support other selection types
                raise ValueError("Only SELECT_ALL selections are supported for scalar datasets")
            if sel.shape != ():
                raise ValueError("Selection shape does not match dataset shape")
        else:
            dims = tuple(shape_json["dims"])
            if sel.shape != dims:
                raise ValueError("Selection shape does not match dataset shape")
        rank = len(dims)  
            
        dtype = self.getDtype(dset_id)
        if self.reader:
            arr = self.reader.getDatasetValues(dset_id, sel)
        else:
            # TBD: Initialize with fill value if non-zero
            arr = np.zeros(sel.shape, dtype=dtype)

        if "updates" in dset_json:
            # apply any non-flushed changes that intersect the current selection
            updates = dset_json["updates"]
            for (update_sel, update_val) in updates:
                sel_inter = selections.intersect(sel, update_sel)
                if sel_inter.nselect == 0:
                    continue
                # update portion of arr, that intersects update_val
                slices = []
                for dim in range(rank):
                    start = sel_inter.start[dim] - sel.start[dim]
                    stop = start + sel_inter.count[dim]
                    slices.append(slice(start, stop, 1))
                slices = tuple(slices)
                arr[slices] = update_val

        return arr
    
    def setDatasetValues(self, dset_id, sel, arr):
        """
        Write the given ndarray to the dataset using the selection
        """
        dset_json = self.getObjectById(dset_id)
        shape_json = dset_json["shape"]
        if not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")
        if sel.select_type not in (selections.H5S_SELECT_HYPERSLABS, selections.H5S_SELECT_ALL):
            # TBD: support other selection types
            raise ValueError("Only hyperslab selections are currently supported")
        if not isinstance(arr, np.ndarray):
            raise TypeError("Expected ndarray for data value")
        if shape_json["class"] == "H5S_NULL":
            raise ValueError("writing to null space dataset not supported")
        if shape_json["class"] == "H5S_SCALAR":
            if sel.shape != ():
                raise ValueError("Selection shape does not match dataset shape")
            if len(arr.shape) > 0:
                raise TypeError("Expected scalar ndarray for scalar dataset")
        else:
            dims = tuple(shape_json["dims"])
            if sel.shape != dims:
                raise ValueError("Selection shape does not match dataset shape")
        if "updates" not in dset_json or sel.select_type == selections.H5S_SELECT_ALL:
            # for select all, throw out any existing updates since this will overwrite them
            dset_json["updates"] = []
        updates = dset_json["updates"]
        updates.append((sel, arr.copy()))
        self.make_dirty(dset_id)


    def resizeDataset(self, dset_id, shape):
        """
        Resize existing Dataset
        """
        self.log.info(f"resizeDataset {dset_id}, {shape}")
        
        dset_json = self.getObjectById(dset_id)  # will throw exception if not found
        if resize_dataset(dset_json, shape):
            self._dirty_objects.add(dset_id)
         

    def deleteObject(self, obj_id):
        """ Delete the given object """
        self.log.info(f"deleteObject: {obj_id}")
        if obj_id not in self.db:
            raise KeyError(f"Object {obj_id} not found for deletion")
        if obj_id == self.root_id:
            raise KeyError("Root group cannot be deleted")
        self.db[obj_id] = None
        
        if obj_id in self._new_objects:
            self._new_objects.remove(obj_id)

        if obj_id in self._dirty_objects:
            self._dirty_objects.remove(obj_id)

        
    def getLinks(self, grp_id):
        """ Get the links for the given group """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            raise KeyError(f"No links - {grp_id} not a group?")
        links = grp_json["links"]
        names = []
        for name in links:
            if links[name] != None:
                names.append(name)
        return names
      
    def getLink(self, grp_id, name):
        """ Get the given link """
        
        obj_json = self.getObjectById(grp_id)
        links = obj_json["links"]
        if name not in links:
            self.log.info(f"Link [{name}] not found in {grp_id}")
            return None
        if links[name] == None:
            self.log.info(f"Link {name} in {grp_id} has been deleted")
            return None

        return links[name]
    
    def _addLink(self, grp_id, name, link_json):
        obj_json = self.getObjectById(grp_id)
        links = obj_json["links"]
        links[name] = link_json
        self.make_dirty(grp_id)
    
    def createHardLink(self, grp_id, name, tgt_id):
        """ Create a new hardlink """
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        link_json["created"] = time.time()
        self._addLink(grp_id, name, link_json)

    def createSoftLink(self, grp_id, name, h5path):
        """ Create a soft link """
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": h5path}
        link_json["created"] = time.time()
        self._addLink(grp_id, name, link_json)

    def createCustomLink(self, grp_id, name, link_json):
        """ create a custom link """
        if link_json.get("class") != "H5L_TYPE_USER_DEFINED":
            link_json["class"] = "H5L_TYPE_USER_DEFINED"
        link_json["created"] = time.time()
        self._addLink(grp_id, name, link_json)

    def createExternalLink(self, grp_id, name, h5path, filepath):
        """ Create a external link link """
        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": h5path, "file": filepath}
        link_json["created"] = time.time()
        self._addLink(grp_id, name, link_json)
 
    def deleteLink(self, grp_id, name):
        """ Delete the given link """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            raise KeyError(f"No links - {grp_id} not a group?")
        links = grp_json["links"]
        if name not in links:
            raise KeyError(f"Link [{name}] not found in {grp_id}")
        links[name] = None  # mark for deletion
        self.make_dirty(grp_id)
 

    def createGroup(self, cpl=None):
        """ Create a new group """

        grp_id = createObjId("groups", root_id=self.root_id)
        group_json = {"attributes": {}, "links": {}}
        if cpl:
            group_json["cpl"] = cpl
        else:
            group_json["cpl"] = {}
        group_json["created"] = time.time()
        self.db[grp_id] = group_json
        self._new_objects.add(grp_id)
        return grp_id
    

    def createCommittedType(self, datatype, cpl=None):
        """
        createCommittedType - creates new named datatype
        Returns item
        """
        self.log.info("createCommittedType")
        if cpl is None:
            cpl = {}
         
        ctype_id = createObjId(obj_type="datatypes", root_id=self.root_id)
        if isinstance(datatype, np.dtype):
            dt = datatype
        else:
            dt = createDataType(datatype)

        type_json = getTypeItem(dt)  # get canonical json description of datatype

        ctype_json = {"type": type_json, "attributes": {}, "cpl": cpl}
        ctype_json["created"] = time.time()
        self.db[ctype_id] = ctype_json
        self._new_objects.add(ctype_id)
        return ctype_id
  
    
    def createDataset(
        self,
        shape=None,
        dtype=None,
        cpl=None,
    ):
        """
        createDataset - creates new dataset given shape and datatype
        Returns obj_id
        """
        type_json = getTypeItem(dtype)
        if shape == "H5S_NULL":
            shape_json = {"class": "H5S_NULL"}
        else:
            shape_json = {"class": "H5S_SIMPLE"}
            shape_json["dims"] = list(shape)

        dset_json = {"shape": shape_json, "type": type_json, "attributes": {}}
        if cpl:
            dset_json["cpl"] = cpl
        else:
            dset_json["cpl"] = {}
 
        dset_id = createObjId("datasets", root_id=self.root_id)   
        self.db[dset_id] = dset_json 
        self._new_objects.add(dset_id)
        return dset_id

    def getCollection(self, col_type=None):
        obj_ids = []
        for obj_id in self.db:
            if self.db[obj_id] == None:
                # skip deleted objects
                continue
            if not col_type or getCollectionForId(obj_id) == col_type:
                obj_ids.append(obj_id)
        return obj_ids

    def __len__(self):
        # return the number of objects
        count = 0
        for obj_id in self.db:
            # skip deleted objects
            if self.db[obj_id] != None:
                count += 1
        return count

    def __iter__(self):
        """ Iterate over object ids """

        for obj_id in self.db:
            if self.db[obj_id] == None:
                # skip deleted objects
                continue
            yield obj_id


    def __contains__(self, obj_id):
        """ Test if a obj id  exists """
        return obj_id in self.db and self.db[obj_id] != None
