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
import logging
import numpy as np
from h5json import Hdf5db
from h5json.hsdsstore.hsds_reader import HSDSReader
from h5json import selections


class HSDSReaderTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(HSDSReaderTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        handler = logging.FileHandler("./hsds_reader_test.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)

    def testSimple(self):
        filepath = "/home/test_user1/test/tall.h5"
        kwargs = {"app_logger": self.log}
        with Hdf5db(**kwargs) as db:
            hsds_reader = HSDSReader(filepath, **kwargs)
            db.reader = hsds_reader
            root_id = db.getObjectIdByPath("/")
            root_json = db.getObjectById(root_id)

            root_attrs = root_json["attributes"]
            self.assertEqual(len(root_attrs), 2)
            self.assertEqual(list(root_attrs.keys()), ["attr1", "attr2"])
            root_links = root_json["links"]
            self.assertEqual(len(root_links), 2)
            self.assertEqual(list(root_links.keys()), ["g1", "g2"])
            g1_link = root_links["g1"]
            self.assertEqual(g1_link["class"], "H5L_TYPE_HARD")
            g1_id = g1_link["id"]
            self.assertEqual(g1_id, db.getObjectIdByPath("/g1/"))
            dset111_id = db.getObjectIdByPath("/g1/g1.1/dset1.1.1")
            dset_json = db.getObjectById(dset111_id)
            dset_type = dset_json["type"]
            self.assertEqual(dset_type["class"], "H5T_INTEGER")
            self.assertEqual(dset_type["base"], "H5T_STD_I32BE")
            dset_attrs = dset_json["attributes"]
            self.assertEqual(len(dset_attrs), 2)
            self.assertEqual(list(dset_attrs.keys()), ["attr1", "attr2"])
            dset_shape = dset_json["shape"]
            self.assertEqual(dset_shape["class"], "H5S_SIMPLE")
            self.assertEqual(dset_shape["dims"], [10, 10])

            # got the 5th row of the dataset
            sel_row = selections.select((10, 10), (5, slice(0, 10)))
            row = db.getDatasetValues(dset111_id, sel_row)
            self.assertTrue(isinstance(row, np.ndarray))
            self.assertEqual(row.shape, (10,))
            for i in range(10):
                v = row[i]
                self.assertEqual(v, i * 5)

            sel_all = selections.select((10, 10), ...)
            arr = db.getDatasetValues(dset111_id, sel_all)
            self.assertTrue(isinstance(arr, np.ndarray))
            self.assertEqual(arr.shape, (10, 10))
            for i in range(10):
                for j in range(10):
                    v = arr[i, j]
                    self.assertEqual(v, i * j)

            # try adding an attribute
            db.createAttribute(dset111_id, "attr3", value=42)
            dset_json = db.getObjectById(dset111_id)
            dset_attrs = dset_json["attributes"]
            self.assertEqual(len(dset_attrs), 3)
            self.assertEqual(list(dset_attrs.keys()), ["attr1", "attr2", "attr3"])
            attr3_json = dset_attrs["attr3"]
            attr3_shape = attr3_json["shape"]
            self.assertEqual(attr3_shape["class"], "H5S_SCALAR")
            attr3_type = attr3_json["type"]
            self.assertEqual(attr3_type["class"], "H5T_INTEGER")
            self.assertEqual(attr3_type["base"], "H5T_STD_I64LE")
            attr3_value = attr3_json["value"]
            self.assertEqual(attr3_value, 42)

            db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
