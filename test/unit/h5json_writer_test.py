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
from os.path import getsize
import logging
import numpy as np
from h5json import Hdf5db
from h5json.jsonstore.h5json_writer import H5JsonWriter

from h5json.hdf5dtype import special_dtype, Reference
from h5json import selections


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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        self.assertTrue(db.writer.lastModified is None)  # no update yet
        db.close()
        self.assertTrue(db.writer.lastModified > 0)  # timestamp should be updated

    def testNullSpaceAttribute(self):

        filepath = "test/unit/out/h5json_writer_testNullSpaceAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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

    def testIntAttribute(self):
        filepath = "test/unit/out/h5json_writer_testIntAttribute.json"

        db = Hdf5db(app_logger=self.log)
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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

    def testCommittedType(self):
        filepath = "test/unit/out/h5json_writer_testCommittedType.json"

        db = Hdf5db(app_logger=self.log)
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
        db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
            db.writer = H5JsonWriter(filepath, data_limit=data_limit, **kwargs)
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


if __name__ == "__main__":
    # setup test files

    unittest.main()
