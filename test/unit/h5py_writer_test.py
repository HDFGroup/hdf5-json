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
import os

import h5py
import numpy as np
from h5json import Hdf5db
from h5json.jsonstore.h5json_reader import H5JsonReader
from h5json.h5pystore.h5py_writer import H5pyWriter
from h5json.hdf5dtype import special_dtype, Reference
from h5json.objid import isRootObjId, isSchema2Id
from h5json import selections


class H5pyWriterTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(H5pyWriterTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        # create logger

        handler = logging.FileHandler("./h5pywritertest.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        # self.log.propagate = False  # prevent log out going to stdout
        self.log.info("init!")

    def testOpen(self):
        filepath = "test/unit/out/h5py_writer_test_testOpen.h5"
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath)
        root_id = db.open()
        self.assertTrue(isSchema2Id(root_id))
        self.assertTrue(isRootObjId(root_id))
        self.assertFalse(db.closed)
        self.assertEqual(db.getObjectIdByPath("/"), root_id)
        db.close()
        self.assertTrue(db.closed)
        obj_id = db.open()
        self.assertEqual(obj_id, root_id)
        db.close()

    def testSimple(self):

        filepath = "test/unit/out/h5py_writer_test_testSimple.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
        self.assertEqual(db.getObjectIdByPath("/"), root_id)
        db.createAttribute(root_id, "attr1", value=[1, 2, 3, 4])
        db.createAttribute(root_id, "attr2", 42)
        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        db.createAttribute(g1_id, "a1", "hello")
        db.close()

        # open file with h5py and verify changes
        with h5py.File(filepath) as f:
            self.assertTrue("attr1", f.attrs)
            self.assertTrue("attr2", f.attrs)
            self.assertEqual(len(f), 1)
            self.assertTrue("g1" in f)
            g1 = f["g1"]
            self.assertTrue("a1" in g1.attrs)
            self.assertEqual(len(g1), 0)
        db.open()

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
        db.close()

        # open file with h5py and verify changes
        with h5py.File(filepath) as f:
            self.assertTrue("attr1", f.attrs)
            self.assertTrue("attr2", f.attrs)
            self.assertEqual(len(f), 2)
            self.assertTrue("g1" in f)
            self.assertTrue("g2" in f)
            g1 = f["g1"]
            self.assertEqual(len(g1), 1)
            self.assertTrue("a1" in g1.attrs)
            self.assertTrue("g1.1" in g1)
            g11 = g1["g1.1"]
            self.assertTrue("dset1.1.1" in g11)
            dset = g11["dset1.1.1"]
            self.assertEqual(dset.shape, (10, 10))
            for i in range(10):
                for j in range(10):
                    self.assertEqual(dset[i, j], i * j)
            self.assertTrue("g2" in f)
            g2 = f["g2"]
            self.assertTrue("extlink" in g2)
            self.assertTrue("slink" in g2)

        db.open()
        db.createAttribute(g1_id, "a1", "hello")
        db.createAttribute(g1_id, "a2", "bye-bye")
        self.assertEqual(len(db.getAttributes(g1_id)), 2)
        db.close()

        with h5py.File(filepath) as f:
            g1 = f["g1"]
            self.assertEqual(len(g1.attrs), 2)
            self.assertTrue("a1" in g1.attrs)
            self.assertTrue("a2" in g1.attrs)

        db.open()
        # test deleting an attribute
        db.deleteAttribute(g1_id, "a1")
        self.assertEqual(len(db.getAttributes(g1_id)), 1)
        self.assertEqual(db.getAttribute(g1_id, "a1"), None)
        db.close()

        with h5py.File(filepath) as f:
            g1 = f["g1"]
            self.assertEqual(len(g1.attrs), 1)
            self.assertFalse("a1" in g1.attrs)
            self.assertTrue("a2" in g1.attrs)

        db.open()
        g21 = db.createGroup()
        db.createHardLink(g2_id, "g2.1", g21)
        db.close()

        with h5py.File(filepath) as f:
            g2 = f["g2"]
            self.assertTrue("g2.1" in g2)

        # create a link, then delete before flushing
        db.open()
        tmp_grp_id = db.createGroup("tmp_group")
        db.createHardLink(g2_id, "tmp_group", tmp_grp_id)
        del_link = db.getLink(g2_id, "tmp_group")
        self.assertTrue(del_link is not None)
        db.deleteLink(g2_id, "tmp_group")
        self.assertEqual(db.getLink(g2_id, "tmp_group"), None)

        db.close()

        with h5py.File(filepath) as f:
            g2 = f["g2"]
            self.assertFalse("tmp_group" in g2)

        db.open()
        sel = selections.select((10, 10), (slice(4, 5), slice(4, 5)))
        arr = np.zeros((), dtype=np.int32)
        arr[()] = 42
        db.setDatasetValues(dset_111_id, sel, arr)
        db.close()

        with h5py.File(filepath) as f:
            dset = f["/g1/g1.1/dset1.1.1"]
            for i in range(10):
                for j in range(10):
                    if i == 4 and j == 4:
                        # this is the one element that was updated
                        expected = 42
                    else:
                        expected = i * j
                    self.assertEqual(dset[i, j], expected)

    def testNullSpaceAttribute(self):

        filepath = "test/unit/out/h5py_writer_test_testNullSpaceAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
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

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            self.assertEqual(f.attrs["A1"], h5py.Empty(dtype=np.int32))

    def testScalarAttribute(self):

        filepath = "test/unit/out/h5py_writer_test_testNullScalarAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
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

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            self.assertTrue(isinstance(a1, np.int32))
            self.assertEqual(a1, 42)

    def testFixedStringAttribute(self):

        filepath = "test/unit/out/h5py_writer_test_testFixedStringAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
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
        db.close()

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            self.assertTrue(isinstance(a1, bytes))
            self.assertEqual(a1, b'Hello, world!')

    def testVlenAsciiAttribute(self):

        filepath = "test/unit/out/h5py_writer_test_testVlenAsciiAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        value = b"Hello, world!"

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
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

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            self.assertTrue(isinstance(a1, str))
            self.assertEqual(a1, value.decode("ascii"))

    def testVlenUtf8Attribute(self):

        filepath = "test/unit/out/h5py_writer_test_testVlenUtf8Attribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        value = "one: \u4e00"

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
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
        self.assertEqual(item["value"], value)
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        db.close()

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            self.assertTrue(isinstance(a1, str))
            self.assertEqual(a1, value)

    def testIntAttribute(self):

        filepath = "test/unit/out/h5py_writer_test_testIntAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        value = [2, 3, 5, 7, 11]

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
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

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            self.assertTrue(isinstance(a1, np.ndarray))
            self.assertEqual(a1.shape, (5,))
            for i in range(5):
                self.assertEqual(a1[i], value[i])

    def testCreateReferenceAttribute(self):

        filepath = "test/unit/out/h5py_writer_test_testCreateReferenceAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
        dset_id = db.createDataset(shape=(), dtype=np.int32)
        db.createHardLink(root_id, "DS1", dset_id)
        dt = special_dtype(ref=Reference)
        ds1_ref = "datasets/" + dset_id
        value = [ds1_ref,]
        db.createAttribute(root_id, "A1", value, dtype=dt)
        attr = db.getAttribute(root_id, "A1")
        self.assertTrue("shape" in attr)
        attr_type = attr["type"]
        self.assertEqual(attr_type["class"], "H5T_REFERENCE")
        self.assertEqual(attr_type["base"], "H5T_STD_REF_OBJ")
        attr_value = db.getAttributeValue(root_id, "A1")
        self.assertEqual(len(attr_value), 1)
        self.assertEqual(attr_value[0], ds1_ref.encode('ascii'))
        db.close()

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            obj_ref = a1[0]
            obj = f[obj_ref]
            self.assertEqual(obj.name, "/DS1")

    def testCreateVlenReferenceAttribute(self):

        filepath = "test/unit/out/h5py_writer_test_testVlenReferenceAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
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

        with h5py.File(filepath) as f:
            self.assertTrue("DS1" in f)
            ds1 = f["DS1"]
            self.assertTrue(ds1)
            self.assertTrue("G1" in f)
            g1 = f["G1"]
            self.assertTrue(g1)
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            ref_obj = f[a1[0]]
            self.assertEqual(ref_obj.name, "/DS1")

    def testCommittedType(self):

        filepath = "test/unit/out/h5py_writer_test_testCommittedType.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        dt = np.dtype("S15")

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
        ctype_id = db.createCommittedType(dt)
        db.createHardLink(root_id, "ctype", ctype_id)
        item = db.getObjectById(ctype_id)
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        db.createHardLink(root_id, "T1", ctype_id)
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

        with h5py.File(filepath) as f:
            self.assertTrue("T1" in f)
            t1 = f["T1"]
            self.assertTrue(isinstance(t1, h5py.Datatype))
            self.assertEqual(t1.dtype, dt)

            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            self.assertEqual(a1, b"hello world!")

    def testCommittedCompoundType(self):

        filepath = "test/unit/out/h5py_writer_test_testCommittedCompoundType.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
        dt_str = special_dtype(vlen=str)
        fields = []
        fields.append(("field_1", np.dtype(">i8")))
        fields.append(("field_2", np.dtype(">f8")))
        fields.append(("field_3", np.dtype("S15")))
        fields.append(("field_4", dt_str))
        dt = np.dtype(fields)
        ctype_id = db.createCommittedType(dt)
        db.createHardLink(root_id, "ctype", ctype_id)
        item = db.getObjectById(ctype_id)
        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        db.createHardLink(root_id, "T1", ctype_id)
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
        arr = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(arr, np.ndarray))
        db.close()

        with h5py.File(filepath) as f:
            self.assertTrue("T1" in f)
            t1 = f["T1"]
            self.assertTrue(isinstance(t1, h5py.Datatype))
            self.assertEqual(len(t1.dtype), 4)
            sub_dt = t1.dtype["field_1"]
            self.assertEqual(sub_dt, np.dtype(">i8"))
            sub_dt = t1.dtype["field_2"]
            self.assertEqual(sub_dt, np.dtype(">f8"))
            sub_dt = t1.dtype["field_3"]
            self.assertEqual(sub_dt, np.dtype("S15"))
            sub_dt = t1.dtype["field_4"]
            self.assertEqual(sub_dt, h5py.special_dtype(vlen=str))

    def testReaderWithUpdate(self):

        file_in = "data/json/tall.json"
        file_out = "test/unit/out/h5py_writer_test_testReaderWithUpdate.h5"
        if os.path.isfile(file_out):
            os.remove(file_out)  # cleanup any previous run

        db = Hdf5db(app_logger=self.log)
        db.reader = H5JsonReader(file_in)
        db.writer = H5pyWriter(file_out)
        db.open()
        # close should create everything the json reader read to the output file
        db.close()
        self.assertTrue(db.closed)

        with h5py.File(file_out) as f:
            self.assertTrue("/g1/g1.1/dset1.1.1" in f)
            dset111 = f["/g1/g1.1/dset1.1.1"]
            self.assertEqual(len(dset111.attrs), 2)

        db.open()
        self.assertFalse(db.closed)
        dset111_id = db.getObjectIdByPath("/g1/g1.1/dset1.1.1")
        db.createAttribute(dset111_id, "attr3", "hello")
        self.assertFalse(db.closed)
        db.close()

        with h5py.File(file_out) as f:
            self.assertTrue("/g1/g1.1/dset1.1.1" in f)
            dset111 = f["/g1/g1.1/dset1.1.1"]
            self.assertEqual(len(dset111.attrs), 3)
            self.assertEqual(dset111.attrs["attr3"], b"hello")

        db.open()
        db.createAttribute(dset111_id, "attr3", "bye-bye")
        db.close()

        with h5py.File(file_out) as f:
            self.assertTrue("/g1/g1.1/dset1.1.1" in f)
            dset111 = f["/g1/g1.1/dset1.1.1"]
            self.assertEqual(len(dset111.attrs), 3)
            self.assertEqual(dset111.attrs["attr3"], b"bye-bye")
            g1 = f["g1"]

        db.open()
        # create a new group
        g13_id = db.createGroup()
        g1_id = db.getObjectIdByPath("/g1")
        db.createHardLink(g1_id, "g1.3", g13_id)
        db.close()

        with h5py.File(file_out) as f:
            g1 = f["g1"]
            self.assertEqual(len(g1), 3)
            self.assertTrue("g1.3" in g1)

        db.open()
        # create a new dataset
        dset_id = db.createDataset(shape=(10, 10), dtype=np.int32)
        db.createHardLink(g1_id, "DS1", dset_id)
        db.close()

        with h5py.File(file_out) as f:
            g1 = f["g1"]
            self.assertTrue("DS1" in g1)
            ds1 = g1["DS1"]
            self.assertEqual(ds1.shape, (10, 10))

        db.open()
        arr = np.asarray(range(10), dtype=np.int32)
        sel = selections.select((10, 10), (slice(5, 6), slice(0, 10)))
        db.setDatasetValues(dset_id, sel, arr)
        db.close()

        with h5py.File(file_out) as f:
            ds1 = f["/g1/DS1"]
            data = ds1[:, :]
            for i in range(10):
                for j in range(10):
                    if i == 5:
                        self.assertEqual(data[i, j], j)
                    else:
                        self.assertEqual(data[i, j], 0)

    def testCompression(self):

        filepath = "test/unit/out/h5py_writer_test_testCompression.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
        self.assertEqual(db.getObjectIdByPath("/"), root_id)
        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)

        layout = {"class": "H5D_CHUNKED", "dims": (10, 1)}
        gzip_filter = {
            "class": "H5Z_FILTER_DEFLATE",
            "id": 1,
            "level": 9,
            "name": "deflate",
        }
        cpl = {"layout": layout, "filters": [gzip_filter, ]}
        dset_id = db.createDataset(shape=(10, 10), dtype=np.int32, cpl=cpl)
        arr = np.zeros((10, 10), dtype=np.int32)
        for i in range(10):
            for j in range(10):
                arr[i, j] = i * j
        sel_all = selections.select((10, 10), ...)
        db.setDatasetValues(dset_id, sel_all, arr)
        db.createHardLink(g1_id, "dset1.1.1", dset_id)
        db.close()

        # open file with h5py and verify changes
        with h5py.File(filepath) as f:

            self.assertTrue("g1" in f)

            g1 = f["g1"]
            self.assertEqual(len(g1), 1)
            self.assertTrue("dset1.1.1" in g1)
            dset = g1["dset1.1.1"]
            self.assertEqual(dset.shape, (10, 10))
            for i in range(10):
                for j in range(10):
                    self.assertEqual(dset[i, j], i * j)


if __name__ == "__main__":
    # setup test files

    unittest.main()
