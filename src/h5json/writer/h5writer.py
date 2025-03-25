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

    @abstractmethod
    def flush(self):
        """ Write dirty items """
        pass

    @abstractmethod
    def close(self):
        """ close storage handle """
        pass
