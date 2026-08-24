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
from abc import ABC, abstractmethod
import weakref

import logging
import time

from .objid import createObjId


class StoragePlugin(ABC):
    """
    This abstract class defines properties and methods that the Hdf5db class uses to read from and
    write to a storage medium.  A single plugin instance is both the reader and the writer for a
    given Hdf5db - there's no separate "reader" object with its own view of the store, so a read
    always reflects whatever the same plugin instance has most recently flushed.
    """

    def __init__(
        self,
        filepath,
        append=False,
        no_data=False,
        read_only=False,
        app_logger=None
    ):
        self._filepath = filepath
        self._append = append
        self._no_data = no_data
        # read_only=True means this plugin must never write to its storage,
        # even if the db it's attached to is otherwise flushed/closed
        # normally (e.g. a source db in a format-conversion tool, which
        # should never risk modifying its input). Concrete plugins should
        # make flush() a safe no-op (raising only if there's actually
        # something pending to write) when this is set, and open() should
        # use the least-privileged access mode the backend supports.
        self._read_only = read_only
        self._db_ref = None
        self._lastModified = None
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

    def set_db(self, db):
        self._db_ref = weakref.ref(db)
        self.log.debug("plugin set db ref")

    @property
    def db(self):
        if not self._db_ref:
            raise ValueError("db not available")
        return self._db_ref()

    @property
    def filepath(self):
        """ return filepath """
        return self._filepath

    @property
    def closed(self):
        """ return True if the plugin's storage handle is closed (or never opened) """
        return self.isClosed()

    @property
    def lastModified(self):
        return self._lastModified

    @property
    def append(self):
        return self._append

    @property
    def no_data(self):
        return self._no_data

    @property
    def read_only(self):
        return self._read_only

    @abstractmethod
    def get_root_id(self):
        """ Return root id """
        pass

    @abstractmethod
    def getObjectById(self, obj_id, include_attrs=True, include_links=True):
        """ return object with given id """
        pass

    @abstractmethod
    def getAttribute(self, obj_id, name, includeData=True):
        """
        Get attribute given an object id and name
        returns: JSON object
        """
        pass

    @abstractmethod
    def getDatasetValues(self, obj_id, sel, dtype=None, query=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """
        pass

    def queryDataset(self, obj_id, query, sel=None, limit=0, update_value=None):
        """
        Query the given dataset using the selection and query expression.

        If update_value is provided, elements matching the query (up to limit elements if limit is
        non-zero) are updated to the given value.

        Return a numpy array of indices for the elements that match the query.
        Plugins are not required to implement this — by default it raises
        NotImplementedError, and Hdf5db falls back to querying the dataset values
        it fetches via getDatasetValues. Override this only if the storage backend
        has a more efficient way to evaluate the query (e.g. pushing it down to storage).
        """
        raise NotImplementedError("queryDataset not implemented for " + type(self).__name__)

    @abstractmethod
    def open(self):
        """ Open storage handle, return root_id """
        pass

    @abstractmethod
    def flush(self):
        """ Write dirty items """
        pass

    @abstractmethod
    def close(self):
        """ close any open handles to the storage """
        pass

    @abstractmethod
    def isClosed(self):
        """ return True if handle is closed """
        pass

    @abstractmethod
    def getStats(self):
        """ return a dictionary object with at minimum the following keys:
            'created': creation time
            'lastModified': modificationTime
            'owner': owner name
        """
        pass

    @abstractmethod
    def getFilters(self, compressors_only=False):
        """ returns a list of filters supported by the plugin """
        pass


class NullPlugin(StoragePlugin):
    """
    This class can be used by HDF5DB as a default no-op plugin - it can't actually read or persist
    anything, but lets a fresh, backend-less Hdf5db still mint a root id and be opened/closed.
    """

    def __init__(
        self,
        filepath,
        append=False,
        no_data=False,
        app_logger=None
    ):
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

        super().__init__(filepath, append=append, no_data=no_data, app_logger=app_logger)
        self.log.debug("NullPlugin.__init__")

        self._root_id = None
        self._is_closed = True

    def get_root_id(self):
        """ Return root id """
        return self._root_id

    def getObjectById(self, obj_id, include_attrs=True, include_links=True):
        """ return object with given id """

        if obj_id != self._root_id:
            raise KeyError(f"{obj_id} not found")

        # create a root group with no links or attributes
        group_json = {"links": {}, "attributes": {}, "creationProperties": {}}
        group_json["created"] = time.time()

        return group_json

    def getAttribute(self, obj_id, name, includeData=True):
        """
        Get attribute given an object id and name
        returns: JSON object
        """
        return None

    def getDatasetValues(self, obj_id, sel=None, dtype=None, query=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """

        # just return None

        return None

    def open(self):
        """ Open storage handle, return root_id """
        self.log.debug("NullPlugin open")
        if self.db is None:
            # no db set yet
            self.log.warning("no self.db db_ref")
            raise ValueError("no db")

        if self._is_closed:
            if not self._root_id:
                if self.db.root_id:
                    # use the db root id
                    self._root_id = self.db.root_id
                else:
                    # create a new root id
                    self._root_id = createObjId(obj_type="groups")
            self._is_closed = False
        return self._root_id

    def flush(self):
        """ Write dirty items """
        self.log.debug("NullPlugin flush")
        # Null plugin is unable to actually persist anything, so return False
        return False

    def close(self):
        """ close any open handles to the storage """
        self._is_closed = True

    def isClosed(self):
        """ return True if handle is closed """
        return self._is_closed

    def getStats(self):
        """ return a dictionary object with at minimum the following keys:
            'created': creation time
            'lastModified': modificationTime
            'owner': owner name
        """
        stats = {}
        stats['created'] = 0
        stats["lastModified"] = 0
        stats['owner'] = ""
        return stats

    def getFilters(self, compressors_only=False):
        """ return empty list of filters """
        return ()
