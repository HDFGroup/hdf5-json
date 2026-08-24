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
import os
import json
from os.path import getsize
from os import stat
import logging
import numpy as np
from h5json import Hdf5db
from h5json.jsonstore.h5json_plugin import H5JsonPlugin

from h5json.hdf5dtype import special_dtype, Reference, RegionReference
from h5json.objid import getUuidFromId
from h5json import selections

# fixture/output paths below are relative to the repo root - normalize cwd so
# this file runs correctly whether invoked from the repo root or from within
# test/unit itself
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


class H5JsonWriterTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(H5JsonWriterTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        # create logger

        handler = logging.FileHandler("./hdf5dbtest.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        # self.log.propagate = False  # prevent log out going to stdout
        self.log.info("init!")

    def testSimple(self):

        filepath = "test/unit/out/h5json_writer_testSimple.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        self.assertEqual(db.getObjectIdByPath("/"), root_id)
        db.createAttribute(root_id, "attr1", value=[1, 2, 3, 4])
        db.createAttribute(root_id, "attr2", 42)
        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
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
        self.assertTrue(db.plugin.lastModified is None)  # no update yet
        db.close()
        self.assertTrue(db.plugin.lastModified > 0)  # timestamp should be updated

    def testNullSpaceAttribute(self):

        filepath = "test/unit/out/h5json_writer_testNullSpaceAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        db.createAttribute(root_id, "A1", None, shape="H5S_NULL", dtype=np.int32)
        item = db.getAttribute(root_id, "A1")
        self.assertTrue("shape" in item)
        shape_item = item["shape"]
        self.assertTrue("class" in shape_item)
        self.assertEqual(shape_item["class"], "H5S_NULL")
        self.assertTrue(item["created"] > time.time() - 1.0)
        value = db.getAttributeValue(root_id, "A1")
        self.assertEqual(value, None)
        db.close()

    def testScalarAttribute(self):
        filepath = "test/unit/out/h5json_writer_testScalarAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        dims = ()
        value = 42
        db.createAttribute(root_id, "A1", value, shape=dims, dtype=np.int32)
        item = db.getAttribute(root_id, "A1")
        shape_json = item["shape"]
        self.assertEqual(shape_json["class"], "H5S_SCALAR")
        self.assertEqual(len(shape_json.keys()), 1)  # just one key should be returned
        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_INTEGER")
        self.assertEqual(item_type["base"], "H5T_STD_I32LE")
        self.assertEqual(len(item_type.keys()), 2)  # just two keys should be returned
        self.assertEqual(item["value"], 42)
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        shape = item["shape"]
        self.assertEqual(shape["class"], "H5S_SCALAR")

        self.assertEqual(item_type["class"], "H5T_INTEGER")
        self.assertEqual(item_type["base"], "H5T_STD_I32LE")
        db.close()

    def testFixedStringAttribute(self):
        filepath = "test/unit/out/h5json_writer_testFixedStringAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        value = "Hello, world!"
        db.createAttribute(root_id, "A1", value, dtype=np.dtype("S13"))  # dims, datatype, value)
        item = db.getAttribute(root_id, "A1")
        shape_json = item["shape"]
        self.assertEqual(shape_json["class"], "H5S_SCALAR")
        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_STRING")
        self.assertEqual(item_type["strPad"], "H5T_STR_NULLPAD")
        self.assertEqual(item_type["length"], 13)
        self.assertEqual(item_type["charSet"], "H5T_CSET_ASCII")
        self.assertEqual(item["value"], "Hello, world!")
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        ret_value = db.getAttributeValue(root_id, "A1")
        self.assertEqual(ret_value, b'Hello, world!')
        db.close()

    def testVlenAsciiAttribute(self):
        filepath = "test/unit/out/h5json_writer_testVlenAsciiAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        value = b"Hello, world!"
        dt = special_dtype(vlen=bytes)

        # write the attribute
        db.createAttribute(root_id, "A1", value, dtype=dt)
        # read it back
        item = db.getAttribute(root_id, "A1")
        shape_json = item["shape"]
        self.assertEqual(shape_json["class"], "H5S_SCALAR")
        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_STRING")
        self.assertEqual(item_type["strPad"], "H5T_STR_NULLTERM")
        self.assertEqual(item_type["length"], "H5T_VARIABLE")
        self.assertEqual(item_type["charSet"], "H5T_CSET_ASCII")
        self.assertEqual(item["value"], "Hello, world!")
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        db.close()

    def testVlenUtf8Attribute(self):
        filepath = "test/unit/out/h5json_writer_testVlenutf8Attribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        value = b"Hello, world!"
        dt = special_dtype(vlen=str)

        # write the attribute
        db.createAttribute(root_id, "A1", value, dtype=dt)
        # read it back
        item = db.getAttribute(root_id, "A1")
        shape_json = item["shape"]
        self.assertEqual(shape_json["class"], "H5S_SCALAR")
        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_STRING")
        self.assertEqual(item_type["strPad"], "H5T_STR_NULLTERM")
        self.assertEqual(item_type["length"], "H5T_VARIABLE")
        self.assertEqual(item_type["charSet"], "H5T_CSET_UTF8")
        self.assertEqual(item["value"], "Hello, world!")
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        db.close()

    def testVlenUtf8AttributeInvalidUtf8(self):
        # createAttribute() fails fast (via array_util.validateUtf8()) on a
        # UTF8-charset string value that can't actually be encoded as valid
        # UTF-8 (a lone surrogate here) - confirms this is enforced on the
        # H5JsonPlugin backend too, not just via NullPlugin. Nothing gets
        # written to the file at all.
        filepath = "test/unit/out/h5json_writer_testVlenUtf8AttributeInvalidUtf8.json"
        if os.path.isfile(filepath):
            os.remove(filepath)

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        dt = special_dtype(vlen=str)
        with self.assertRaises(UnicodeEncodeError):
            db.createAttribute(root_id, "A1", "\udc80abc", dtype=dt)

        self.assertIsNone(db.getAttribute(root_id, "A1"))
        db.close()

        with open(filepath) as f:
            data = json.load(f)
        root_json = data["groups"][data["root"]]
        self.assertNotIn("attributes", root_json)

    def testIntAttribute(self):
        filepath = "test/unit/out/h5json_writer_testIntAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        value = [2, 3, 5, 7, 11]
        db.createAttribute(root_id, "A1", value, dtype=np.int16)
        item = db.getAttribute(root_id, "A1")
        self.assertEqual(item["value"], [2, 3, 5, 7, 11])
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        item_shape = item["shape"]
        self.assertEqual(item_shape["class"], "H5S_SIMPLE")
        self.assertEqual(item_shape["dims"], [5,])
        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_INTEGER")
        self.assertEqual(item_type["base"], "H5T_STD_I16LE")
        db.close()

    def testCreateReferenceAttribute(self):
        filepath = "test/unit/out/h5json_writer_testCreateReferenceAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        dset_id = db.createDataset(shape=(), dtype=np.int32)
        db.createHardLink(root_id, "DS1", dset_id)

        dt = special_dtype(ref=Reference)

        ds1_ref = "datasets/" + dset_id
        value = [ds1_ref,]
        db.createAttribute(root_id, "A1", value, dtype=dt)
        attr = db.getAttribute(root_id, "A1")
        self.assertTrue("shape" in attr)
        shape = attr["shape"]
        self.assertEqual(shape["class"], "H5S_SIMPLE")
        self.assertEqual(shape["dims"], [1,])

        attr_type = attr["type"]
        self.assertEqual(attr_type["class"], "H5T_REFERENCE")
        self.assertEqual(attr_type["base"], "H5T_STD_REF_OBJ")
        attr_value = attr["value"]
        self.assertEqual(len(attr_value), 1)
        self.assertEqual(attr_value[0], ds1_ref)
        db.close()

    def testCreateVlenReferenceAttribute(self):
        filepath = "test/unit/out/h5json_writer_testVlenReferenceAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        dset_id = db.createDataset(shape=(), dtype=np.int32)
        db.createHardLink(root_id, "DS1", dset_id)
        grp_id = db.createGroup()
        db.createHardLink(root_id, "G1", grp_id)

        dt_base = special_dtype(ref=Reference)
        dt = special_dtype(vlen=dt_base)

        ds1_ref = "datasets/" + dset_id
        grp_ref = "groups/" + grp_id
        ref_arr = np.zeros((2,), dtype=dt_base)
        ref_arr[0] = ds1_ref
        ref_arr[1] = grp_ref
        vlen_arr = np.zeros((), dtype=dt)
        vlen_arr[()] = ref_arr

        db.createAttribute(root_id, "A1", vlen_arr)
        item = db.getAttribute(root_id, "A1")

        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_VLEN")
        self.assertEqual(item_type["size"], "H5T_VARIABLE")
        base_type = item_type["base"]
        self.assertEqual(base_type["class"], "H5T_REFERENCE")
        self.assertEqual(base_type["base"], "H5T_STD_REF_OBJ")

        item_shape = item["shape"]
        self.assertEqual(item_shape["class"], "H5S_SCALAR")
        db.close()

    def testCreateOpaqueAttribute(self):
        # matches the format used in data/json/opaque_attr.json:
        # {"value": "<base64>", "encoding": "base64"}
        filepath = "test/unit/out/h5json_writer_testCreateOpaqueAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        dt = np.dtype("V2")
        value = np.zeros((), dtype=dt)
        value[()] = b'\xfe\xff'
        db.createAttribute(root_id, "A1", value, dtype=dt)

        attr = db.getAttribute(root_id, "A1")
        self.assertEqual(attr["type"], {"class": "H5T_OPAQUE", "size": 2})
        self.assertEqual(attr["value"], "/v8=")
        self.assertEqual(attr["encoding"], "base64")
        db.close()

    def testCreateOpaqueDataset(self):
        # matches the format used in data/json/opaque_dset.json
        filepath = "test/unit/out/h5json_writer_testCreateOpaqueDataset.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        dt = np.dtype("V2")
        shape = (4,)
        dset_id = db.createDataset(shape=shape, dtype=dt)
        db.createHardLink(root_id, "DS1", dset_id)
        arr = np.zeros(shape, dtype=dt)
        arr[3] = b'\xfe\xff'
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)

        dumped = db.plugin.dumpDataset(dset_id)
        self.assertEqual(dumped["type"], {"class": "H5T_OPAQUE", "size": 2})
        self.assertEqual(dumped["value"], ["", "", "", "/v8="])
        self.assertEqual(dumped["encoding"], "base64")
        db.close()

    def testCreateRegionReferenceAttribute(self):
        # matches the format used in data/json/regionref_attr.json:
        # {"id": <bare uuid>, "select_type": ..., "selection": [...]}
        filepath = "test/unit/out/h5json_writer_testCreateRegionReferenceAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        target_id = db.createDataset(shape=(3, 16), dtype=np.int32)
        db.createHardLink(root_id, "DS1", target_id)

        sel = selections.select((3, 16), ([0, 2, 1, 2], [1, 11, 0, 4]))
        ref = RegionReference("datasets/" + target_id, sel)

        dt = special_dtype(ref=RegionReference)
        value = np.empty((1,), dtype=dt)
        value[0] = ref.tobytes()

        db.createAttribute(root_id, "A1", value, dtype=dt)
        attr = db.getAttribute(root_id, "A1")
        attr_type = attr["type"]
        self.assertEqual(attr_type["class"], "H5T_REFERENCE")
        self.assertEqual(attr_type["base"], "H5T_STD_REF_DSETREG")

        attr_value = attr["value"]
        self.assertEqual(len(attr_value), 1)
        self.assertEqual(attr_value[0]["id"], getUuidFromId(target_id))
        self.assertEqual(attr_value[0]["select_type"], "H5S_SEL_POINTS")
        self.assertEqual(attr_value[0]["selection"], [[0, 1], [2, 11], [1, 0], [2, 4]])
        db.close()

    def testCreateRegionReferenceDataset(self):
        # matches the format used in data/json/regionref_dset.json
        filepath = "test/unit/out/h5json_writer_testCreateRegionReferenceDataset.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        target_id = db.createDataset(shape=(3, 16), dtype=np.int32)
        db.createHardLink(root_id, "DS1", target_id)

        sel = selections.select((3, 16), (slice(0, 2), slice(0, 4)))
        ref = RegionReference("datasets/" + target_id, sel)

        dt = special_dtype(ref=RegionReference)
        ref_dset_id = db.createDataset(shape=(1,), dtype=dt)
        db.createHardLink(root_id, "DS2", ref_dset_id)

        ref_arr = np.empty((1,), dtype=dt)
        ref_arr[0] = ref.tobytes()
        sel_all = selections.select((1,), ...)
        db.setDatasetValues(ref_dset_id, sel_all, ref_arr)

        dumped = db.plugin.dumpDataset(ref_dset_id)
        dumped_type = dumped["type"]
        self.assertEqual(dumped_type["class"], "H5T_REFERENCE")
        self.assertEqual(dumped_type["base"], "H5T_STD_REF_DSETREG")

        value = dumped["value"]
        self.assertEqual(len(value), 1)
        self.assertEqual(value[0]["id"], getUuidFromId(target_id))
        self.assertEqual(value[0]["select_type"], "H5S_SEL_HYPERSLABS")
        self.assertEqual(value[0]["selection"], [[[0, 0], [1, 3]]])
        db.close()

    def testCommittedType(self):
        filepath = "test/unit/out/h5json_writer_testCommittedType.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        dt = np.dtype("S15")

        ctype_id = db.createCommittedType(dt)
        db.createHardLink(root_id, "ctype", ctype_id)
        item = db.getObjectById(ctype_id)
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)

        item_type = item["type"]

        self.assertEqual(item_type["class"], "H5T_STRING")
        self.assertEqual(item_type["strPad"], "H5T_STR_NULLPAD")
        self.assertEqual(item_type["charSet"], "H5T_CSET_ASCII")
        self.assertEqual(item_type["length"], 15)

        # create an attribute using the committed type
        db.createAttribute(root_id, "A1", "hello world!", dtype=f"datatypes/{ctype_id}")
        attr = db.getAttribute(root_id, "A1")
        self.assertEqual(attr["value"], "hello world!")

        attr_type = attr["type"]
        self.assertEqual(attr_type["class"], "H5T_STRING")
        self.assertEqual(attr_type["length"], 15)
        self.assertEqual(attr_type["charSet"], "H5T_CSET_ASCII")
        db.close()

    def testCommittedCompoundType(self):
        filepath = "test/unit/out/h5json_writer_testCommittedCompoundType.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        dt_str = special_dtype(vlen=str)
        fields = []
        fields.append(("field_1", np.dtype(">i8")))
        fields.append(("field_2", ">f8"))
        fields.append(("field_3", np.dtype("S15")))
        fields.append(("field_4", dt_str))
        dt = np.dtype(fields)

        ctype_id = db.createCommittedType(dt)
        db.createHardLink(root_id, "ctype", ctype_id)
        item = db.getObjectById(ctype_id)
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)

        item_type = item["type"]

        self.assertEqual(item_type["class"], "H5T_COMPOUND")
        fields = item_type["fields"]
        self.assertEqual(len(fields), 4)

        # create an attribute using the committed type
        attr_value = (42, 3.14, "circle", "area = R^2 * PI")
        db.createAttribute(root_id, "A1", attr_value, dtype=f"datatypes/{ctype_id}")
        attr = db.getAttribute(root_id, "A1")
        self.assertEqual(attr["value"], list(attr_value))
        attr_shape = attr["shape"]
        self.assertEqual(attr_shape["class"], "H5S_SCALAR")

        attr_type = attr["type"]
        self.assertEqual(attr_type["class"], "H5T_COMPOUND")

        value = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(value, np.ndarray))
        db.close()

    def testNoData(self):

        def init_db(db):
            root_id = db.getObjectIdByPath("/")
            db.createAttribute(root_id, "attr1", value=[1, 2, 3, 4])
            db.createAttribute(root_id, "attr2", 42)
            g1_id = db.createGroup()
            db.createHardLink(root_id, "g1", g1_id)
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
            dset_0_id = db.createDataset(shape=(), dtype=np.int32)
            arr = np.zeros((), dtype=np.int32)
            arr[()] = 42
            sel_all = selections.select((), ...)
            db.setDatasetValues(dset_0_id, sel_all, arr)
            db.createHardLink(g1_1_id, "dset1.1.1", dset_111_id)
            db.createHardLink(g1_1_id, "dset0", dset_0_id)
            db.createSoftLink(g2_id, "slink", "somewhere")
            db.createExternalLink(g2_id, "extlink", "somewhere", "someplace")
            db.createCustomLink(g2_id, "cust", {"foo": "bar"})

        def save_json(filepath, data_limit=None):
            db = Hdf5db(app_logger=self.log)
            kwargs = {"indent": 2, "app_logger": self.log}
            db.plugin = H5JsonPlugin(filepath, data_limit=data_limit, **kwargs)
            db.open()
            init_db(db)
            db.close()
            file_size = getsize(filepath)
            return file_size

        file_prefix = "test/unit/out/h5json_writer_testNoData_"

        size_with_data = save_json(file_prefix + "withData.json", data_limit=None)
        # should be close to 4640
        self.assertTrue(size_with_data > 4000)

        size_without_data = save_json(file_prefix + "withoutData.json", data_limit=0)
        # should be close to 3038
        self.assertTrue(size_without_data > 3000)
        self.assertTrue(size_without_data < 4000)

        size_with_smalldata = save_json(file_prefix + "withSmallData.json", data_limit=100)
        # should be close to 3057
        self.assertTrue(size_with_smalldata > size_without_data)
        self.assertTrue(size_with_smalldata < size_with_data)

    def testDumpGroupCreationProperties(self):
        filepath = "test/unit/out/h5json_writer_testDumpGroupCreationProperties.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        cpl = {"linkCreationOrder": "H5P_CRT_ORDER_TRACKED"}
        g1_id = db.createGroup(cpl=cpl)
        db.createHardLink(root_id, "g1", g1_id)

        dumped = db.plugin.dumpGroup(g1_id)
        self.assertTrue("creationProperties" in dumped)
        self.assertEqual(dumped["creationProperties"], cpl)

        db.close()

    def testGetStats(self):
        # exercises H5JsonWriter.getStats(), which Hdf5db itself never calls
        filepath = "test/unit/out/h5json_writer_testGetStats.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()
        db.createAttribute(root_id, "attr1", 42)
        db.close()  # flush so the file actually exists on disk

        stats = db.plugin.getStats()
        self.assertEqual(set(stats.keys()), {"created", "lastModified", "owner"})

        os_stat_info = stat(filepath)
        self.assertEqual(stats["created"], os_stat_info.st_ctime)
        self.assertEqual(stats["lastModified"], os_stat_info.st_mtime)
        self.assertEqual(stats["owner"], os_stat_info.st_uid)

    def testGetFilters(self):
        # exercises H5JsonWriter.getFilters(), which always returns an empty
        # tuple since the json store doesn't implement compression filters
        filepath = "test/unit/out/h5json_writer_testGetFilters.json"

        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        db.open()

        self.assertEqual(db.plugin.getFilters(), ())
        self.assertEqual(db.plugin.getFilters(compressors_only=True), ())

        db.close()

    def testMultipleFlushesPreserveEarlierData(self):
        # regression test: H5JsonWriter used to only ever dump the file once
        # (guarded by a "_file_dumped" flag) - a second flush() was a no-op.
        # That guard was removed to support Hdf5db's periodic auto-flush, but
        # naively re-dumping the ENTIRE db from scratch on every flush() call
        # is itself unsafe: Hdf5db.getDatasetValues() can only reconstruct a
        # dataset's value from its pending (not yet flushed) update or from a
        # reader - once an earlier flush() clears that dataset's pending
        # update, a later full re-derive (with no reader attached) would
        # incorrectly return a zero-filled array, corrupting already-flushed
        # data. dumpGroups()/dumpDatasets()/dumpDatatypes() now only
        # recompute entries for objects that are new/dirty/resized since the
        # previous flush, leaving unchanged (already-flushed) entries alone.
        filepath = "test/unit/out/h5json_writer_testMultipleFlushesPreserveEarlierData.json"

        db = Hdf5db(app_logger=self.log, auto_flush_memory=None, auto_flush_interval=None)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        shape = (10,)
        arr1 = np.arange(10, dtype=np.int32)
        dset1_id = db.createDataset(shape, dtype=np.int32)
        db.createHardLink(root_id, "dset1", dset1_id)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset1_id, sel_all, arr1)

        # first flush persists dset1 and clears its pending update
        db.flush()

        # write and flush a second, unrelated dataset
        arr2 = np.arange(10, 20, dtype=np.int32)
        dset2_id = db.createDataset(shape, dtype=np.int32)
        db.createHardLink(root_id, "dset2", dset2_id)
        db.setDatasetValues(dset2_id, sel_all, arr2)
        db.flush()

        # a third (redundant) flush with nothing new should also be harmless
        db.flush()
        db.close()

        rdb = Hdf5db(app_logger=self.log)
        rdb.plugin = H5JsonPlugin(filepath, app_logger=self.log, read_only=True)
        rdb.open()
        result1 = rdb.getDatasetValues(rdb.getObjectIdByPath("/dset1"), sel_all)
        result2 = rdb.getDatasetValues(rdb.getObjectIdByPath("/dset2"), sel_all)
        self.assertTrue(np.array_equal(result1, arr1))
        self.assertTrue(np.array_equal(result2, arr2))
        rdb.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
