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
import os
import unittest
import logging
import numpy as np
from h5json import Hdf5db
from h5json.jsonstore.h5json_plugin import H5JsonPlugin
from h5json.hdf5dtype import special_dtype, RegionReference
from h5json.objid import getUuidFromId
from h5json import selections

# fixture/output paths below are relative to the repo root - normalize cwd so
# this file runs correctly whether invoked from the repo root or from within
# test/unit itself
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


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
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
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

    def testOpaqueAttribute(self):
        # reads the actual fixture file the opaque JSON format was designed
        # around
        filepath = "data/json/opaque_attr.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")
        attr = db.getAttribute(ds1_id, "A1")
        self.assertEqual(attr["type"], {"class": "H5T_OPAQUE", "size": 7})
        value = attr["value"]
        self.assertEqual(len(value), 4)
        self.assertEqual(value[0], 'T1BBUVVFMA==')
        self.assertEqual(value[1], 'T1BBUVVFMQ==')
        self.assertEqual(value[2], 'T1BBUVVFMg==')
        self.assertEqual(value[3], 'T1BBUVVFMw==')
        self.assertEqual(attr["encoding"], "base64")

        value = db.getAttributeValue(ds1_id, "A1")
        self.assertEqual(value.dtype, np.dtype("V7"))
        self.assertEqual(value.tobytes(), b'OPAQUE0OPAQUE1OPAQUE2OPAQUE3')

        db.close()

    def testOpaqueDataset(self):
        # reads the actual fixture file the opaque JSON format was designed
        # around
        filepath = "data/json/opaque_dset.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")
        sel_all = selections.select((4,), ...)
        arr = db.getDatasetValues(ds1_id, sel_all)
        self.assertEqual(arr.dtype, np.dtype("V7"))
        self.assertEqual(arr.shape, (4,))
        self.assertEqual(
            [v.tobytes() for v in arr],
            [b"OPAQUE0", b"OPAQUE1", b"OPAQUE2", b"OPAQUE3"],
        )

        db.close()

    def testDatasetCreationProperties(self):
        filepath = "data/json/fillvalue.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        db.open()

        dset_id = db.getObjectIdByPath("/dset")
        dset_json = db.getObjectById(dset_id)
        self.assertTrue("creationProperties" in dset_json)
        self.assertEqual(dset_json["creationProperties"]["fillValue"], 42)

        db.close()

    def testAttributeIncludeData(self):
        filepath = "data/json/opaque_attr.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")

        attr = db.plugin.getAttribute(ds1_id, "A1", includeData=True)
        self.assertTrue("value" in attr)

        attr = db.plugin.getAttribute(ds1_id, "A1", includeData=False)
        self.assertFalse("value" in attr)
        self.assertFalse("encoding" in attr)
        self.assertEqual(attr["type"], {"class": "H5T_OPAQUE", "size": 7})

        db.close()

    def testArrayDataset(self):
        # reads a real fixture with an H5T_ARRAY (subarray) dtype - a
        # regression test for jsonToArray() not accounting for the subarray
        # dims when sanity-checking the constructed array's size/shape
        # (used to raise "setting an array element with a sequence")
        filepath = "data/json/array_dset.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")
        sel_all = selections.select((4,), ...)
        arr = db.getDatasetValues(ds1_id, sel_all)
        self.assertEqual(arr.shape, (4, 3, 5))
        expected = [
            [[0, 0, 0, 0, 0], [0, -1, -2, -3, -4], [0, -2, -4, -6, -8]],
            [[0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [2, 1, 0, -1, -2]],
            [[0, 2, 4, 6, 8], [2, 3, 4, 5, 6], [4, 4, 4, 4, 4]],
            [[0, 3, 6, 9, 12], [3, 5, 7, 9, 11], [6, 7, 8, 9, 10]],
        ]
        self.assertTrue(np.array_equal(arr, np.array(expected)))

        db.close()

    def testRegionReferenceAttribute(self):
        # reads the actual fixture file the region reference JSON format was
        # designed around
        filepath = "data/json/regionref_attr.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
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
        wdb.plugin = H5JsonPlugin(write_path, app_logger=self.log)
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
        rdb.plugin = H5JsonPlugin(write_path, app_logger=self.log, read_only=True)
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

    def testGetRootId(self):
        # exercises H5JsonReader.get_root_id(), which Hdf5db itself never
        # calls directly - it uses the return value of reader.open() instead
        filepath = "data/json/tall.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        # before open() the reader hasn't parsed the file yet, so its own
        # root id is still unset
        self.assertIsNone(db.plugin.get_root_id())

        root_id = db.open()
        self.assertEqual(db.plugin.get_root_id(), root_id)
        self.assertEqual(db.plugin.get_root_id(), db.root_id)
        db.close()

    def testGetDtype(self):
        # exercises H5JsonReader.getDtype(), used internally by
        # getDatasetValues()/getAttribute() consumers but not otherwise
        # invoked directly against the reader
        filepath = "data/json/tall.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        db.open()

        dset111_id = db.getObjectIdByPath("/g1/g1.1/dset1.1.1")
        dset_json = db.plugin.getObjectById(dset111_id)
        dtype = db.plugin.getDtype(dset_json)
        self.assertEqual(dtype, np.dtype(">i4"))

        # a datatype item without a "type" key should raise KeyError
        with self.assertRaises(KeyError):
            db.plugin.getDtype({"shape": {"class": "H5S_SCALAR"}})

        db.close()

    def testGetDtypeCommittedType(self):
        # getDtype() also has to resolve a "datatypes/<uuid>" reference to a
        # committed type - exercise that branch via a round trip through
        # H5JsonWriter/H5JsonReader
        write_path = "test/unit/out/h5json_reader_testGetDtypeCommittedType.json"

        wdb = Hdf5db(app_logger=self.log)
        wdb.plugin = H5JsonPlugin(write_path, app_logger=self.log)
        root_id = wdb.open()
        dt = np.dtype("S15")
        ctype_id = wdb.createCommittedType(dt)
        wdb.createHardLink(root_id, "ctype", ctype_id)
        wdb.createAttribute(root_id, "A1", "hello world!", dtype=f"datatypes/{ctype_id}")
        wdb.close()

        rdb = Hdf5db(app_logger=self.log)
        rdb.plugin = H5JsonPlugin(write_path, app_logger=self.log, read_only=True)
        rdb.open()
        root_id2 = rdb.getObjectIdByPath("/")
        attr_json = rdb.plugin.getAttribute(root_id2, "A1")
        resolved_dtype = rdb.plugin.getDtype(attr_json)
        self.assertEqual(resolved_dtype, dt)
        rdb.close()

    def testGetStats(self):
        # exercises H5JsonReader.getStats(), which Hdf5db itself never calls
        filepath = "data/json/tall.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        db.open()

        stats = db.plugin.getStats()
        self.assertEqual(set(stats.keys()), {"created", "lastModified", "owner"})

        file_stat = os.stat(filepath)
        self.assertEqual(stats["created"], file_stat.st_ctime)
        self.assertEqual(stats["lastModified"], file_stat.st_mtime)
        self.assertEqual(stats["owner"], file_stat.st_uid)

        db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
