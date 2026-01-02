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

import numpy as np
import logging
from .hdf5dtype import getTypeItem, createDataType, Reference, special_dtype
from .hdf5dtype import numpy_integer_types, numpy_float_types
from .array_util import jsonToArray, bytesArrayToList
from .dset_util import resize_dataset, getDatasetLayoutClass
from .shape_util import getShapeClass, getShapeDims, getShapeJson
from .filters import validateFilters
from .objid import createObjId, getCollectionForId, isValidUuid, getUuidFromId, getHashTagForId
from . import selections
from .time_util import getNow
from .apiversion import _apiver
from .h5reader import H5Reader, H5NullReader
from .h5writer import H5Writer, H5NullWriter


def _decode(item, encoding="ascii"):
    """
    decode any byte items to python 3 strings
    """
    ret_val = None
    if type(item) is bytes:
        ret_val = item.decode(encoding)
    elif type(item) is list:
        ret_val = []
        for x in item:
            ret_val.append(_decode(x, encoding))
    elif type(item) is tuple:
        ret_val = []
        for x in item:
            ret_val.append(_decode(x, encoding))
        ret_val = tuple(ret_val)
    elif type(item) is dict:
        ret_val = {}
        for k in dict:
            ret_val[k] = _decode(item[k], encoding)
    elif type(item) is np.ndarray:
        x = item.tolist()
        ret_val = []
        for x in item:
            ret_val.append(_decode(x, encoding))
    elif type(item) in numpy_integer_types:
        ret_val = int(item)
    elif type(item) in numpy_float_types:
        ret_val = float(item)
    else:
        ret_val = item
    return ret_val


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
        app_logger=None,
    ):
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

        self._db = {}

        self._new_objects = set()       # set of for newly created objects
        self._dirty_objects = set()     # set of modified objects
        self._deleted_objects = set()   # set of deleted objects
        self._resized_datasets = set()  # set of dataset ids that have been resized
        self._dataset_updates = {}         # list of dataset values updates keyed by dset_id

        self._root_id = None

        if h5_reader:
            self._reader = h5_reader
            self._reader.set_db(self)
        else:
            self._reader = None

        if h5_writer:
            self._writer = h5_writer
            self._writer.set_db(self)
        else:
            self._writer = None

    @property
    def db(self):
        """ return object db dictionary """
        return self._db

    @property
    def reader(self):
        """ return reader instance """
        return self._reader

    @reader.setter
    def reader(self, value: H5Reader):
        """ set the reader """
        if self._writer and not self._writer.isClosed():
            self.flush()
        if self._reader and not self._reader.isClosed():
            self._reader.close()
        self._reader = value
        self._reader.set_db(self)

    @property
    def writer(self):
        """ return writer instance """
        return self._writer

    @writer.setter
    def writer(self, value: H5Writer):
        """ set the writer """
        if self._writer:
            self._writer.close()
        self._writer = value
        if self._writer:
            self._writer.set_db(self)

    @property
    def root_id(self):
        """ return root uuid """
        return self._root_id

    def is_new(self, obj_id):
        """ return true if this is a new object (has not been persisted) """
        obj_id = getHashTagForId(obj_id)
        return obj_id in self._new_objects

    def is_dirty(self, obj_id):
        """ return true if this object has been modified """
        obj_id = getHashTagForId(obj_id)
        if self.is_new(obj_id):
            return True
        if obj_id in self._resized_datasets:
            return True
        return obj_id in self._dirty_objects

    def is_resized(self, dset_id):
        """ return true if this dataset has been resized """
        dset_id = getHashTagForId(dset_id)

        if dset_id in self._resized_datasets:
            return True
        else:
            return False

    @property
    def new_objects(self):
        return self._new_objects

    @property
    def dirty_objects(self):
        return self._dirty_objects

    @property
    def deleted_objects(self):
        return self._deleted_objects

    @property
    def resized_datasets(self):
        return self._resized_datasets

    def _getDatasetUpdates(self, dset_id):
        """ Get list of update tuples """
        if getCollectionForId(dset_id) != "datasets":
            raise TypeError("expected dataset id")
        if dset_id not in self._dataset_updates:
            self._dataset_updates[dset_id] = []
        return self._dataset_updates[dset_id]

    def make_dirty(self, obj_id):
        """ Mark the object as dirty and update the lastModified timestamp """
        obj_id = getHashTagForId(obj_id)
        if obj_id not in self.db:
            self.log.error("make dirty called on deleted object")
            raise KeyError(f"obj_id: {obj_id} not found")
        if self.db[obj_id] is None:
            # object deleted, just return
            return
        obj_json = self.db[obj_id]
        obj_json["lastModified"] = getNow()
        if not self.is_new(obj_id):
            # object hasn't been initially written yet, add to dirty_object set
            self._dirty_objects.add(obj_id)

    def flush(self):
        """ write out any changes """
        self.log.debug("db.flush()")
        self._checkWriter()
        if not self.writer.flush():
            # flush not successful, don't clear dirty set
            self.log.error("writer flush failed")
            return False

        # reset new, dirty and deleted sets
        self._new_objects.clear()
        self._dirty_objects.clear()
        self._deleted_objects.clear()
        self._resized_datasets.clear()
        self._dataset_updates.clear()
        return True

    def readAll(self):
        """ read all meta data objects from reader and save to db """

        self.log.debug("readAll")
        if self.closed:
            raise IOError("database is not open")

        obj_ids = set()
        obj_ids.add(self.root_id)
        while obj_ids:
            obj_id = obj_ids.pop()
            self.log.debug(f"readAll, get {obj_id}")
            obj_json = self.getObjectById(obj_id)  # will add obj_id to db if not already present
            if getCollectionForId(obj_id) == "groups":
                # add any hard links to the set
                links = obj_json["links"]
                for title in links:
                    link_json = links[title]
                    if "id" in link_json:
                        link_id = link_json["id"]
                        obj_ids.add(link_id)

    def open(self):
        """ open reader and writer if set """
        self.log.debug("db.open()")

        if self.reader is None:
            self.reader = H5NullReader(None, app_logger=self.log)
            self._reader.set_db(self)

        if self.writer is None:
            self.writer = H5NullWriter(None, app_logger=self.log)
            self._writer.set_db(self)

        if not self.reader.isClosed():
            self.log.debug("db is already opened")
            raise IOError("db is already opened")

        if self.writer.append:
            # append mode for the writer, first open writer and get the root id
            self.log.debug("db.open, write append, getting root_id from writer")
            writer_root_id = self.writer.open()
            if self._root_id:
                if writer_root_id != self._root_id:
                    raise IOError("writer root id does not match reader root id")
            else:
                self._root_id = writer_root_id

            # now open reader
            reader_root_id = self.reader.open()
            if reader_root_id != self._root_id:
                raise IOError("db root id does not match reader root id")

        else:
            # open reader first and get root id
            reader_root_id = self.reader.open()
            self.log.debug(f"got reader root_id:  {reader_root_id}")

            if self._root_id:
                if reader_root_id != self._root_id:
                    raise IOError("reader root id does not match reader root id")
            else:
                self._root_id = reader_root_id
            self.log.debug("open writer")
            # now open writer
            writer_root_id = self.writer.open()
            self.log.debug(f"got writer root_id: {writer_root_id}")
            if writer_root_id != self._root_id:
                raise IOError("writer root id does not match reader root id")

        self.log.debug(f"db.open() - returning root_id: {self._root_id}")
        return self._root_id

    def close(self):
        """ close reader and writer handles """
        self.log.info("Hdf5db __close")

        if self.writer and not isinstance(self.writer, H5NullWriter):
            self.flush()
            self.writer.close()
        if self.reader:
            self.reader.close()

    @property
    def closed(self):
        if self.reader:
            return self.reader.isClosed()
        elif self.writer:
            return self.writer.isClosed()
        elif self._root_id:
            return True
        else:
            return False

    def __enter__(self):
        """ called on package init """
        self.log.info("Hdf5db __enter")
        return self

    def __exit__(self, type, value, traceback):
        """ called on package exit """
        self.log.info("Hdf5db __exit")
        self.close()

    def _checkReader(self):
        """ check the reader is set and open """
        if self.reader is None:
            raise IOError("reader not set")
        if self.reader.isClosed():
            raise IOError("reader is closed")

    def _checkWriter(self):
        """ check the writer is set and open """
        if self.writer is None:
            raise IOError("writer not set")
        if self.writer.isClosed():
            raise IOError("writer is closed")

    def getObjectById(self, obj_id, refresh=False):
        """ return object with given id """
        self._checkReader()
        obj_id = getHashTagForId(obj_id)
        if obj_id not in self.db or refresh:
            # load the obj from the reader
            self.log.debug(f"getObjectById - fetching {obj_id} from reader")
            obj_json = self.reader.getObjectById(obj_id)
            self.db[obj_id] = obj_json
        obj_json = self.db[obj_id]

        return obj_json

    def getObjectIdByPath(self, h5path, parent_id=None):
        """ Return id for the given link path starting from parent_id if set,
        otherwise the root_id """

        if h5path == "/":
            return self.root_id  # just return root id

        if parent_id is None:
            parent_id = self.root_id
        else:
            parent_id = getHashTagForId(parent_id)

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

    def getDtype(self, obj_json):
        """ Return numpy data type for given dataset, datatype, or attribute
        """

        if "type" not in obj_json:
            # group id?
            raise TypeError(f"{obj_json} does not have a datatype")
        type_item = obj_json["type"]
        if isValidUuid(type_item) and getCollectionForId(type_item) == "datatypes":
            ctype_id = "t-" + getUuidFromId(type_item)
            ctype_json = self.getObjectById(ctype_id)
            if ctype_json is None:
                raise KeyError(f"ctype: {ctype_id} not found")

            type_json = ctype_json["type"].copy()
            type_json["id"] = ctype_id
            dtype = createDataType(type_json)
        else:
            dtype = createDataType(type_item)

        return dtype

    def getAttributes(self, obj_id):
        """
        Get attributes given an object id and name
        returns: JSON object
        """

        obj_json = self.getObjectById(obj_id)
        attrs = obj_json["attributes"]
        names = []

        for name in attrs:
            attr_json = attrs[name]
            if attr_json is None:
                continue
            if "DELETED" in attr_json:
                continue  # deleted attr
            names.append(name)

        return names

    def getAttribute(self, obj_id, name, includeData=True):
        """
        Get attribute given an object id and name
        returns: JSON object
        """

        attr_names = self.getAttributes(obj_id)
        if name not in attr_names:
            return None

        obj_json = self.getObjectById(obj_id)
        attrs = obj_json["attributes"]

        attr_json = attrs[name]

        return attr_json

    def getAttributeValue(self, obj_id, name):
        """ Return NDArray of the given attribute value """
        attr_json = self.getAttribute(obj_id, name)
        if attr_json is None:
            raise KeyError(f"attribute {name} not found")
        shape_json = attr_json["shape"]
        if shape_json["class"] == "H5S_NULL":
            # no value for empty shape attributes
            return None
        elif shape_json["class"] == "H5S_SCALAR":
            dims = ()
        else:
            dims = shape_json["dims"]
        dtype = self.getDtype(attr_json)

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

        # First, make sure we have a NumPy array
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
        type_json = getTypeItem(dtype)
        # finally put it all together...
        attr_json = {"shape": shape_json, "type": type_json, "value": value_json}
        attr_json["created"] = getNow()

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
        attr_json = attrs_json[name]
        attr_json["DELETED"] = getNow()  # mark key for deletion

        self.make_dirty(obj_id)

    def getDatasetValues(self, dset_id, sel):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """

        def init_arr(dtype, cpl):
            """ create an ndarray with the give shape, dtype and fill_value
                (if the latter is found in the creation properties list) """
            arr_shape = sel.count if isinstance(sel.count, tuple) else (sel.count, )
            arr = np.zeros(arr_shape, dtype=dtype)
            if "fillValue" in cpl:
                fillValue = cpl["fillValue"]
                # TBD: fix for compound types
                arr[...] = fillValue
            return arr

        dset_id = getHashTagForId(dset_id)
        self.log.info(f"getDatasetValues dset_id: {dset_id}, sel: {sel}")

        dset_json = self.getObjectById(dset_id)
        shape_json = dset_json["shape"]
        if not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")

        dtype = self.getDtype(dset_json)

        if "creationProperties" in dset_json:
            cpl = dset_json["creationProperties"]
        else:
            cpl = {}

        updates = self._getDatasetUpdates(dset_id)

        shape_class = getShapeClass(shape_json)

        if shape_class == "H5S_NULL":
            # return None for selections on null space
            return None

        if sel.shape != getShapeDims(shape_json):
            raise ValueError("Selection shape does not match dataset shape")

        if shape_class == "H5S_SCALAR":
            if sel.select_type != selections.H5S_SELECT_ALL:
                # TBD: support other selection types
                raise ValueError("Only SELECT_ALL selections are supported for scalar datasets")
            if sel.shape != ():
                raise ValueError("Selection shape does not match dataset shape")
            if updates:
                # for scalars the update has to be the requested value
                (update_sel, arr) = updates[-1]
            elif dset_id in self._new_objects:
                arr = init_arr(dtype, cpl)
            else:
                # fetch from the server
                arr = self.reader.getDatasetValues(dset_id, sel, dtype=dtype)
                if arr is None:
                    raise KeyError(f"Data for dataset {dset_id} not returned")
            # done with NULL and SCALAR cases
            return arr

        # simple daaset
        arr = None
        fetch = True

        # determine if we need to get data from the reader
        if isinstance(self._reader, H5NullReader) or dset_id in self._new_objects:
            fetch = False
        else:
            for (update_sel, update_val) in updates:
                sel_inter = selections.intersect(sel, update_sel)
                if sel_inter.nselect == 0:
                    continue
                if selections.contained(sel, update_sel):
                    # desired selection is wholly contained in this update
                    # TBD: determine if multiple updates would contain all the
                    # required elements
                    fetch = False
                    break
        if fetch:
            # get last saved version of the data from the reader
            arr = self.reader.getDatasetValues(dset_id, sel, dtype=dtype)
        else:
            # initialize an array with fill value if given
            arr = init_arr(dtype, cpl)

        # apply any updates that impact this selection
        for (update_sel, update_val) in updates:
            # get the part of the update that is in common with the requested selection
            x_sel = selections.intersect(sel, update_sel)
            if x_sel.nselect == 0:
                # this update doesn't effect the selection, so ignore
                continue
            # apply the update to the array to be returned
            src_sel = selections.translate(update_sel, x_sel)
            tgt_sel = selections.translate(sel, x_sel)
            arr[tgt_sel.slices] = update_val[src_sel.slices]

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
        tgt_dt = self.getDtype(dset_json)
        src_dt = arr.dtype
        if src_dt != tgt_dt:
            raise TypeError("arr.dtype doesn't match dataset dtype")
        shape_class = getShapeClass(shape_json)
        if shape_class == "H5S_NULL":
            raise ValueError("writing to null space dataset not supported")
        if shape_class == "H5S_SCALAR":
            if sel.shape != ():
                raise ValueError("Selection shape does not match dataset shape")
            if len(arr.shape) > 0:
                raise TypeError("Expected scalar ndarray for scalar dataset")
        else:
            dims = getShapeDims(shape_json)
            if sel.shape != dims:
                raise ValueError("Selection shape does not match dataset shape")
        updates = self._getDatasetUpdates(dset_id)
        if sel.select_type == selections.H5S_SELECT_ALL:
            # for select all, throw out any existing updates since this will overwrite them
            updates.clear()
        arr = arr.copy()  # make a copy in case the client updates it later
        rank = len(sel.shape)
        if len(arr.shape) < rank:
            # reshape to keep compatiblity with dataset rank
            if sel.select_type == selections.H5S_SELECT_ALL:
                # this should not result in a dimension reduction
                raise ValueError("unexpected selection shape")
            if sel.select_type != selections.H5S_SELECT_HYPERSLABS:
                raise ValueError("tbd")
            arr = arr.reshape(sel.mshape)
        updates.append((sel, arr))
        self.make_dirty(dset_id)

    def resizeDataset(self, dset_id, shape):
        """
        Resize existing Dataset
        """
        self.log.info(f"resizeDataset {dset_id}, {shape}")

        dset_json = self.getObjectById(dset_id)  # will throw exception if not found
        old_dims = getShapeDims(dset_json)
        resize_dataset(dset_json, shape)

        if dset_id not in self.new_objects:
            self._resized_datasets.add(dset_id)

        new_dims = getShapeDims(dset_json)
        rank = len(new_dims)

        # adjust any selections in the update list
        updates = self._getDatasetUpdates(dset_id)
        for i in range(len(updates)):
            (sel_update, arr) = updates[i]
            if sel_update.select_type == selections.H5S_SELECT_HYPERSLABS:
                slices = list(sel_update.slices)
                for dim in range(rank):
                    s = slices[dim]
                    if s.stop > new_dims[dim]:
                        # selection outside new bounds of dataset
                        slices[dim] = slice(s.start, new_dims[dim], s.step)
                sel_update = selections.select(new_dims, tuple(slices))
                updates[i] = (sel_update, arr)

        # if the shape has shrunk in any dimension, do a flush now
        do_flush = False
        for i in range(len(new_dims)):
            if new_dims[i] < old_dims[i]:
                do_flush = True
                break

        if do_flush:
            self.flush()

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

        if obj_id in self._resized_datasets:
            self._resized_datasets.remove(obj_id)

        self._deleted_objects.add(obj_id)

    def getLinks(self, grp_id):
        """ Get the links for the given group """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            raise KeyError(f"No links - {grp_id} not a group?")
        links = grp_json["links"]
        names = []
        for name in links:
            link_json = links[name]
            if link_json is None:
                continue
            if "DELETED" in link_json:
                continue  # deleted link
            names.append(name)
        return names

    def getLink(self, grp_id, name):
        """ Get the given link """

        obj_json = self.getObjectById(grp_id)
        links = obj_json["links"]
        if name not in links:
            self.log.info(f"Link [{name}] not found in {grp_id}")
            return None
        link_json = links[name]
        if "DELETED" in link_json:
            self.log.info(f"Link {name} in {grp_id} has been deleted")
            return None

        return link_json

    def _addLink(self, grp_id, name, link_json):
        obj_json = self.getObjectById(grp_id)
        links = obj_json["links"]
        links[name] = link_json
        self.make_dirty(grp_id)

    def createHardLink(self, grp_id, name, tgt_id):
        """ Create a new hardlink """
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def createSoftLink(self, grp_id, name, h5path):
        """ Create a soft link """
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": h5path}
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def createCustomLink(self, grp_id, name, link_json):
        """ create a custom link """
        if link_json.get("class") != "H5L_TYPE_USER_DEFINED":
            link_json["class"] = "H5L_TYPE_USER_DEFINED"
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def createExternalLink(self, grp_id, name, h5path, filepath):
        """ Create a external link link """
        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": h5path, "file": filepath}
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def deleteLink(self, grp_id, name):
        """ Delete the given link """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            raise KeyError(f"No links - {grp_id} not a group?")
        links = grp_json["links"]
        if name not in links:
            raise KeyError(f"Link [{name}] not found in {grp_id}")
        link_json = links[name]
        link_json["DELETED"] = getNow()  # mark for deletion
        self.make_dirty(grp_id)
        grp_json = self.getObjectById(grp_id)
        links = grp_json["links"]

    def createGroup(self, cpl=None):
        """ Create a new group """
        if self.closed:
            raise ValueError("db is closed")
        grp_id = createObjId("groups", root_id=self.root_id)
        group_json = {"attributes": {}, "links": {}}
        if cpl:
            group_json["creationProperties"] = cpl
        else:
            group_json["creationProperties"] = {}
        group_json["created"] = getNow()
        self.db[grp_id] = group_json
        self._new_objects.add(grp_id)
        return grp_id

    def createCommittedType(self, datatype, cpl=None):
        """
        createCommittedType - creates new named datatype
        Returns item
        """
        if self.closed:
            raise ValueError("db is closed")
        self.log.info("createCommittedType")
        if cpl is None:
            cpl = {}

        ctype_id = createObjId(obj_type="datatypes", root_id=self.root_id)
        if isinstance(datatype, np.dtype):
            dt = datatype
        else:
            dt = createDataType(datatype)

        type_json = getTypeItem(dt)  # get canonical json description of datatype

        ctype_json = {"type": type_json, "attributes": {}, "creationProperties": cpl}
        ctype_json["created"] = getNow()
        self.db[ctype_id] = ctype_json
        self._new_objects.add(ctype_id)
        return ctype_id

    def createDataset(
        self,
        shape=None,
        maxdims=None,
        dtype=None,
        cpl=None,
    ):
        """
        createDataset - creates new dataset given shape and datatype
        Returns obj_id
        """
        if self.closed:
            raise ValueError("db is closed")
        type_json = getTypeItem(dtype)
        shape_json = getShapeJson(shape, maxdims=maxdims)

        dset_json = {"shape": shape_json, "type": type_json, "attributes": {}}
        if cpl:
            if "filters" in cpl:
                if self.writer:
                    supported_filters = self.writer.getFilters()
                else:
                    supported_filters = ()
                # validate and normalize supplied filter property list
                validateFilters(cpl["filters"], supported_filters=supported_filters)
            if cpl.get("fillValue"):
                fillvalue = cpl["fillValue"]
                # is it compatible with the array type?
                if hasattr(fillvalue, "tolist"):
                    # convert numpy object to list
                    fillvalue = fillvalue.tolist()
                fillvalue = _decode(fillvalue)
                if not isinstance(fillvalue, str) and hasattr(fillvalue, "__iter__"):
                    # fill value is a list, or similar: check that dtype is compound
                    if len(fillvalue) != len(dtype):
                        raise ValueError("Invalid fill value for non-compound type dataset")
                    fillvalue = list(fillvalue)
                    cpl["fillValue"] = fillvalue
                else:
                    if type_json["class"] == "H5T_COMPOUND":
                        raise ValueError("Invalid fill value for compound type dataset")
            dset_json["creationProperties"] = cpl
        else:
            dset_json["creationProperties"] = {}

        if maxdims and getDatasetLayoutClass(dset_json) != "H5D_CHUNKED":
            raise ValueError("Only datasets with 'H5D_CHUNKED' layout can be resizable")
        dset_json["created"] = getNow()

        dset_id = createObjId("datasets", root_id=self.root_id)
        self.db[dset_id] = dset_json
        self._new_objects.add(dset_id)
        return dset_id

    def getCollection(self, col_type=None):
        obj_ids = []
        for obj_id in self.db:
            if self.db[obj_id] is None:
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
            if self.db[obj_id] is not None:
                count += 1
        return count

    def __iter__(self):
        """ Iterate over object ids """

        for obj_id in self.db:
            if self.db[obj_id] is None:
                # skip deleted objects
                continue
            yield obj_id

    def __contains__(self, obj_id):
        """ Test if a obj id  exists """
        return obj_id in self.db and self.db[obj_id] is not None
