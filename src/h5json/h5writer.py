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
from .objid import createObjId


class H5Writer(ABC):
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
        self._filepath = filepath
        self._append = append
        self._no_data = no_data
        self._filepath = filepath
        self._db_ref = None
        self._lastModified = None
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

    def set_db(self, db):
        self._db_ref = weakref.ref(db)
        self.log.debug("writer set db ref")

    @property
    def filepath(self):
        return self._filepath

    @property
    def closed(self):
        return self.isClosed()

    @property
    def lastModified(self):
        return self._lastModified

    @property
    def db(self):
        if not self._db_ref:
            self.log.debug("db not available")
            return None
        return self._db_ref()

    @property
    def append(self):
        return self._append

    @property
    def no_data(self):
        return self._no_data

    @abstractmethod
    def open(self):
        """ open storage handle, return root_id"""
        pass

    @abstractmethod
    def flush(self):
        """ Write dirty items """
        # return False since we can't actually persist anything
        return False

    @abstractmethod
    def close(self):
        """ close storage handle """
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
        """ returns a list of filters supported by the writer """
        pass


class H5NullWriter(H5Writer):
    """
    This class can be used by HDF5DB as a default no-op writer
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

        if append:
            raise IOError("append is not supported for H5NullWriter")

        super().__init__(filepath, no_data=no_data, app_logger=app_logger)
        self.log.debug("H5NullWriter.__init__")
        self._root_id = None
        self._is_closed = True

    def open(self):
        """ open storage handle, return root_id"""
        self.log.debug("H5NullWriter open")
        if not self._is_closed:
            return self._root_id  # already open

        if self.db is None:
            # no db set yet
            self.log.warning("no self.db db_ref")
            raise ValueError("no db")

        if not self._root_id:
            if self.db.root_id:
                self._root_id = self.db.root_id
            else:
                self._root_id = createObjId(obj_type="groups")
        self._is_closed = False
        return self._root_id

    def flush(self):
        """ Write dirty items """
        self.log.debug("H5NullWriter> flush")
        # Null writer is unable to actually persist anything, so return False
        return False

    def close(self):
        """ close storage handle """
        self.log.debug("H5NullWriter.close")
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
        """ return empty list of filters  """

        return ()
