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
from h5json.jsonstore.h5json_reader import H5JsonReader
from h5json.jsonstore.h5json_writer import H5JsonWriter
from h5json.hdf5dtype import special_dtype, RegionReference
from h5json.objid import getUuidFromId
from h5json import selections


class H5pyReaderTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(H5pyReaderTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        handler = logging.FileHandler("./h5json_reader_test.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)

    def testSimple(self):
        filepath = "data/json/tall.json"
        db = Hdf5db(app_logger=self.log)
        db.reader = H5JsonReader(filepath, app_logger=self.log)
        self.assertTrue(db.closed)
        root_id = db.open()
        self.assertTrue(root_id)
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

    def testRegionReferenceAttribute(self):
        # reads the actual fixture file the region reference JSON format was
        # designed around
        filepath = "data/json/regionref_attr.json"
        db = Hdf5db(app_logger=self.log)
        db.reader = H5JsonReader(filepath, app_logger=self.log)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")
        ds2_id = db.getObjectIdByPath("/DS2")
        ds2_uuid = getUuidFromId(ds2_id)

        value = db.getAttributeValue(ds1_id, "A1")
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value.shape, (2,))
        self.assertEqual(value.dtype.metadata.get("ref"), RegionReference)

        ref0 = RegionReference.frombytes(value[0])
        self.assertEqual(ref0.id, "d-" + ds2_uuid)
        sel0 = selections.Selection.frombytes(ref0.selection_bytes)
        self.assertEqual(sel0.select_type, selections.H5S_SEL_POINTS)
        self.assertEqual(
            ref0.to_json(),
            {
                "id": ds2_uuid,
                "select_type": "H5S_SEL_POINTS",
                "selection": [[0, 1], [2, 11], [1, 0], [2, 4]],
            },
        )

        # second value is a 4-block hyperslab selection in the fixture file -
        # this model expands multi-block hyperslabs into the equivalent point
        # selection on read (see selections.from_region_json())
        ref1 = RegionReference.frombytes(value[1])
        self.assertEqual(ref1.id, "d-" + ds2_uuid)
        sel1 = selections.Selection.frombytes(ref1.selection_bytes)
        self.assertEqual(sel1.select_type, selections.H5S_SEL_POINTS)
        self.assertEqual(sel1.nselect, 32)  # 4 blocks x 8 points each

        db.close()

    def testRegionReferenceWriteReadRoundTrip(self):
        write_path = "test/unit/out/h5json_reader_testRegionReferenceRoundTrip.json"

        wdb = Hdf5db(app_logger=self.log)
        wdb.writer = H5JsonWriter(write_path, app_logger=self.log)
        root_id = wdb.open()

        target_id = wdb.createDataset(shape=(3, 16), dtype=np.int32)
        wdb.createHardLink(root_id, "DS1", target_id)

        sel = selections.select((3, 16), (slice(0, 2), slice(0, 4)))
        ref = RegionReference("datasets/" + target_id, sel)

        dt = special_dtype(ref=RegionReference)
        ref_dset_id = wdb.createDataset(shape=(1,), dtype=dt)
        wdb.createHardLink(root_id, "DS2", ref_dset_id)
        ref_arr = np.empty((1,), dtype=dt)
        ref_arr[0] = ref.tobytes()
        sel_all = selections.select((1,), ...)
        wdb.setDatasetValues(ref_dset_id, sel_all, ref_arr)
        wdb.close()

        rdb = Hdf5db(app_logger=self.log)
        rdb.reader = H5JsonReader(write_path, app_logger=self.log)
        rdb.open()
        read_target_id = rdb.getObjectIdByPath("/DS1")
        read_ref_dset_id = rdb.getObjectIdByPath("/DS2")
        sel_all = selections.select((1,), ...)
        arr = rdb.getDatasetValues(read_ref_dset_id, sel_all)
        self.assertEqual(arr.shape, (1,))

        read_ref = RegionReference.frombytes(arr[0])
        self.assertEqual(read_ref.id, read_target_id)
        read_sel = selections.Selection.frombytes(read_ref.selection_bytes)
        self.assertEqual(read_sel.to_region_json(), sel.to_region_json())
        rdb.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
