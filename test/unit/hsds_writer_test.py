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
import unittest
import time
import logging
import h5py
import numpy as np
from h5json import Hdf5db
from h5json.hsdsstore.hsds_writer import HSDSWriter
from h5json.hdf5dtype import special_dtype, Reference
from h5json import selections


class HSDSWriterTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(HSDSWriterTest, self).__init__(*args, **kwargs)
        # main

        # create logger
        logfname = "hsds_writer_test.log"
        loglevel = logging.DEBUG
        logging.basicConfig(filename=logfname, format='%(levelname)s %(asctime)s %(message)s', level=loglevel)
        self.log = logging.getLogger()
        self.log.info("init!")

    def testSimple(self):

        filepath = "/home/test_user1/writer_test.h5"
        db = Hdf5db(app_logger=self.log)
        db.writer = HSDSWriter(filepath)
        root_id = db.open()
        print("root_id:", root_id)
        db.createAttribute(root_id, "attr1", value=[1, 2, 3, 4])
        db.createAttribute(root_id, "attr2", 42)
        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        db.createAttribute(g1_id, "a1", "hello")
        g2_id = db.createGroup()
        db.createHardLink(root_id, "g2", g2_id)

        g1_1_id = db.createGroup()
        db.createHardLink(g1_id, "g1.1", g1_1_id)
        dset_111_id = db.createDataset(shape=(10, 10), dtype=np.int32)
        arr = np.zeros((10, 10), dtype=np.int32)
        for i in range(10):
            for j in range(10):
                arr[i, j] = i * j
        sel_all = selections.select((10, 10), ...)
        db.setDatasetValues(dset_111_id, sel_all, arr)
        db.createHardLink(g1_1_id, "dset1.1.1", dset_111_id)
        db.createSoftLink(g2_id, "slink", "somewhere")
        db.createExternalLink(g2_id, "extlink", "somewhere", "someplace")
        db.createCustomLink(g2_id, "cust", {"foo": "bar"})
        db.flush()

        db.createAttribute(g1_id, "a2", "bye-bye")
        db.flush()

        g21 = db.createGroup()
        db.createHardLink(g2_id, "g2.1", g21)
        db.flush()

        sel = selections.select((10, 10), (slice(4, 5), slice(4, 5)))
        arr = np.zeros((), dtype=np.int32)
        arr[()] = 42
        db.setDatasetValues(dset_111_id, sel, arr)
        db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
