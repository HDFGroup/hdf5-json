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

        filepath = "test/unit/out/h5json_writer_testSimple.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
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
            db.createHardLink(g1_1_id, "dset1.1.1", dset_111_id)
            db.createSoftLink(g2_id, "slink", "somewhere")
            db.createExternalLink(g2_id, "extlink", "somewhere", "someplace")
            db.createCustomLink(g2_id, "cust", {"foo": "bar"})
            db.flush()

    def testNullSpaceAttribute(self):

        filepath = "test/unit/out/h5json_writer_testNullSpaceAttribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")
            db.createAttribute(root_id, "A1", None, shape="H5S_NULL", dtype=np.int32)
            item = db.getAttribute(root_id, "A1")
            self.assertTrue("shape" in item)
            shape_item = item["shape"]
            self.assertTrue("class" in shape_item)
            self.assertEqual(shape_item["class"], "H5S_NULL")
            self.assertTrue(item["created"] > time.time() - 1.0)
            value = db.getAttributeValue(root_id, "A1")
            self.assertEqual(value, None)

    def testScalarAttribute(self):
        filepath = "test/unit/out/h5json_writer_testScalarAttribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")
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

    def testFixedStringAttribute(self):
        filepath = "test/unit/out/h5json_writer_testFixedStringAttribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")
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

    def testVlenAsciiAttribute(self):
        filepath = "test/unit/out/h5json_writer_testVlenAsciiAttribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")

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

    def testVlenUtf8Attribute(self):
        filepath = "test/unit/out/h5json_writer_testVlenutf8Attribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")

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

    def testIntAttribute(self):
        filepath = "test/unit/out/h5json_writer_testIntAttribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")
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

    def testCreateReferenceAttribute(self):
        filepath = "test/unit/out/h5json_writer_testCreateReferenceAttribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")

            dset_id = db.createDataset(shape=(), dtype=np.int32)
            db.createHardLink(root_id, "DS1", dset_id)

            dt = special_dtype(ref=Reference)

            ds1_ref = "datasets/" + dset_id
            value = [ds1_ref,]
            db.createAttribute(root_id, "A1", value, dtype=dt)
            item = db.getAttribute(root_id, "A1")
            attr = db.getAttribute(root_id, "A1")
            self.assertTrue("shape" in attr)

            attr_type = attr["type"]
            self.assertEqual(attr_type["class"], "H5T_REFERENCE")
            self.assertEqual(attr_type["base"], "H5T_STD_REF_OBJ")
            attr_value = item["value"]
            self.assertEqual(len(attr_value), 1)
            self.assertEqual(attr_value[0], ds1_ref)

    def testCreateVlenReferenceAttribute(self):
        filepath = "test/unit/out/h5json_writer_testVlenReferenceAttribute.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")
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

    def testCommittedType(self):
        filepath = "test/unit/out/h5json_writer_testCommittedType.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")
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

    def testCommittedCompoundType(self):
        filepath = "test/unit/out/h5json_writer_testCommittedCompoundType.h5"

        with Hdf5db(app_logger=self.log) as db:
            db.writer = H5JsonWriter(filepath, app_logger=self.log)
            root_id = db.getObjectIdByPath("/")

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


if __name__ == "__main__":
    # setup test files

    unittest.main()
