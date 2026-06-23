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


class H5Reader(ABC):
    """
    This abstract class defines properties and methods that the Hdf5db class uses for reading from an HDF5
    compatible storage medium.
    """

    def __init__(
        self,
        filepath,
        app_logger=None
    ):
        self._filepath = filepath
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

    def set_db(self, db):
        self._db_ref = weakref.ref(db)

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
        """ return True if the reader handle is closed (or never opened) """
        return self.isClosed()

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
    def getDatasetValues(self, obj_id, sel=None, dtype=None, query=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """
        pass

    @abstractmethod
    def open(self):
        """ Open data source for reading """
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


class H5NullReader(H5Reader):
    """
    This class can be used by HDF5DB as a default no-op reader
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
        self.log.debug("H5NullReader.__init__")

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
        group_json = {"links": {}, "attributes": {}, "cpl": {}}
        group_json["created"] = time.time()

        return group_json

    def getAttribute(self, obj_id, name, includeData=True):
        """
        Get attribute given an object id and name
        returns: JSON object
        """
        return None

    def getDatasetValues(self, obj_id, sel=None, dtype=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """

        # just return None

        return None

    def open(self):
        """ Open data source for reading """
        self.log.debug("H5NullReader open")
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
