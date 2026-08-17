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
from h5json.h5pystore.h5py_reader import H5pyReader
from h5json.h5pystore.h5py_writer import H5pyWriter
from h5py import h5r
from h5json.hdf5dtype import special_dtype, Reference, RegionReference
from h5json.objid import isRootObjId, isSchema2Id, getUuidFromId
from h5json import selections
from h5json.time_util import getNow


class H5pyTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(H5pyTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        # create logger

        handler = logging.FileHandler("./h5pytest.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        # self.log.propagate = False  # prevent log out going to stdout
        self.log.info("init!")

    # --- H5pyReader tests ---

    def testReadTall(self):
        filepath = "data/hdf5/tall.h5"
        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(filepath, app_logger=self.log)
        root_id = db.open()
        root_json = db.getObjectById(root_id)
        root_attrs = root_json["attributes"]
        self.assertEqual(len(root_attrs), 2)
        self.assertEqual(list(root_attrs.keys()), ["attr1", "attr2"])
        root_links = root_json["links"]
        self.assertEqual(len(root_links), 2)
        self.assertEqual(list(root_links.keys()), ["g1", "g2"])
        g1_link = root_links["g1"]
        self.assertEqual(g1_link["class"], "H5L_TYPE_HARD")
        self.assertTrue("created" in g1_link)
        g1_created = g1_link["created"]
        now = getNow()
        self.assertTrue(g1_created < int(now))

        g1_id = g1_link["id"]
        self.assertTrue(g1_id)
        self.assertEqual(g1_id, db.getObjectIdByPath("/g1/"))
        dset111_id = db.getObjectIdByPath("/g1/g1.1/dset1.1.1")
        dset_json = db.getObjectById(dset111_id)
        dset_type = dset_json["type"]
        self.assertEqual(dset_type["class"], "H5T_INTEGER")
        self.assertEqual(dset_type["base"], "H5T_STD_I32BE")
        dset_attrs = dset_json["attributes"]
        self.assertEqual(len(dset_attrs), 2)
        self.assertEqual(list(dset_attrs.keys()), ["attr1", "attr2"])
        attr1_json = dset_attrs["attr1"]
        for k in ("type", "shape", "value", "created"):
            self.assertTrue(k in attr1_json)
        dset_shape = dset_json["shape"]
        self.assertEqual(dset_shape["class"], "H5S_SIMPLE")
        dims = dset_shape["dims"]
        self.assertEqual(dims, [10, 10])
        dims = tuple(dims)

        # read one element from a dataset
        sel = selections.select(dims, (slice(4, 5), slice(5, 6)))
        arr = db.getDatasetValues(dset111_id, sel)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.shape, (1, 1))
        self.assertEqual(arr[0, 0], 20)

        # read one row
        sel = selections.select(dims, (slice(4, 5), slice(0, 10)))
        arr = db.getDatasetValues(dset111_id, sel)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.shape, (1, 10))
        self.assertEqual(list(arr[0]), list(range(0, 40, 4)))

        # do a point selection; dset1.1.1[i,j] = i*j, so diagonals are i*i
        sel = selections.select(dims, [(0, 0), (1, 1), (2, 2), (3, 3)])
        arr = db.getDatasetValues(dset111_id, sel)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.shape, (4,))
        for i in range(4):
            self.assertEqual(arr[i], i * i)

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

    def testQueryReader(self):
        # exercises Hdf5db.queryDataset's reader-backed (ChunkIterator) fallback
        # path against data read from an actual .h5 file, rather than in-memory
        # updates
        filepath = "data/hdf5/tall.h5"
        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(filepath, app_logger=self.log)
        db.open()
        dset111_id = db.getObjectIdByPath("/g1/g1.1/dset1.1.1")

        # dset1.1.1[i, j] = i*j over a 10x10 grid; i*j > 10 for 56 of the 100 elements
        query = "_ > 10"
        indices = db.queryDataset(dset111_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (56, 2))
        for row in indices:
            i, j = int(row[0]), int(row[1])
            self.assertTrue(i * j > 10)

        # restrict to the first row (i=0) - i*j is always 0 there, so no matches
        sel = selections.select((10, 10), (slice(0, 1), slice(0, 10)))
        indices = db.queryDataset(dset111_id, query, sel=sel)
        self.assertEqual(indices.shape, (0, 2))

        # limit to the first 10 matches
        indices = db.queryDataset(dset111_id, query, limit=10)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (10, 2))
        for row in indices:
            i, j = int(row[0]), int(row[1])
            self.assertTrue(i * j > 10)

        db.close()

    def testReadOpaqueDataset(self):
        # reads a real HDF5 file whose opaque data has an actual HDF5 "tag"
        # attached ("Character array") - h5py's high-level API can't read
        # this at all (raises OSError: no appropriate function for
        # conversion path) unless the memory type's tag is set to match, so
        # this also exercises H5pyReader._readOpaqueDataset()'s low-level
        # tag-matched read.
        filepath = "data/hdf5/opaque_dset.h5"
        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(filepath, app_logger=self.log)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")
        sel_all = selections.select((4,), ...)
        arr = db.getDatasetValues(ds1_id, sel_all)
        self.assertEqual(arr.dtype, np.dtype("V7"))
        self.assertEqual(
            [v.tobytes() for v in arr],
            [b"OPAQUE0", b"OPAQUE1", b"OPAQUE2", b"OPAQUE3"],
        )

        # a partial (hyperslab) selection - exercises the numpy-indexing
        # path taken after the full tag-matched read
        sel = selections.select((4,), slice(1, 3))
        arr2 = db.getDatasetValues(ds1_id, sel)
        self.assertEqual([v.tobytes() for v in arr2], [b"OPAQUE1", b"OPAQUE2"])

        db.close()

    def testReadOpaqueAttribute(self):
        # reads a real HDF5 file whose opaque attribute has an actual HDF5
        # "tag" attached - see testReadOpaqueDataset()
        filepath = "data/hdf5/opaque_attr.h5"
        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(filepath, app_logger=self.log)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")
        attr = db.getAttribute(ds1_id, "A1")
        self.assertEqual(attr["type"], {"class": "H5T_OPAQUE", "size": 7})
        self.assertEqual(attr["encoding"], "base64")

        value = db.getAttributeValue(ds1_id, "A1")
        self.assertEqual(value.dtype, np.dtype("V7"))
        self.assertEqual(
            [v.tobytes() for v in value],
            [b"OPAQUE0", b"OPAQUE1", b"OPAQUE2", b"OPAQUE3"],
        )

        db.close()

    def testWriteReadOpaqueRoundTrip(self):
        filepath = "test/unit/out/h5py_test_testWriteReadOpaqueRoundTrip.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        wdb = Hdf5db(app_logger=self.log)
        wdb.writer = H5pyWriter(filepath, no_data=False)
        root_id = wdb.open()

        dt = np.dtype("V2")
        shape = (4,)
        dset_id = wdb.createDataset(shape=shape, dtype=dt)
        wdb.createHardLink(root_id, "DS1", dset_id)
        arr = np.zeros(shape, dtype=dt)
        arr[3] = b'\xfe\xff'
        sel_all = selections.select(shape, ...)
        wdb.setDatasetValues(dset_id, sel_all, arr)

        attr_val = np.zeros((), dtype=dt)
        attr_val[()] = b'\xfe\xff'
        wdb.createAttribute(root_id, "A1", attr_val, dtype=dt)
        wdb.close()

        rdb = Hdf5db(app_logger=self.log)
        rdb.reader = H5pyReader(filepath, app_logger=self.log)
        root_id2 = rdb.open()
        ds1_id = rdb.getObjectIdByPath("/DS1")

        result = rdb.getDatasetValues(ds1_id, sel_all)
        self.assertEqual(result.dtype, dt)
        self.assertEqual([v.tobytes() for v in result], [b'\x00\x00'] * 3 + [b'\xfe\xff'])

        attr_value = rdb.getAttributeValue(root_id2, "A1")
        self.assertEqual(attr_value.dtype, dt)
        self.assertEqual(attr_value.tobytes(), b'\xfe\xff')

        rdb.close()

    def testReadRegionReferenceAttribute(self):
        # reads a real HDF5 file with region-reference attributes.  h5py can
        # resolve which dataset a region reference points to, but there's no
        # generic way to recover its selection back out of the file, so the
        # reader binds each RegionReference to its target dataset only, with
        # no selection (see H5pyReader._copy_element()).
        filepath = "data/hdf5/regionref_attr.h5"
        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(filepath, app_logger=self.log)
        db.open()

        ds1_id = db.getObjectIdByPath("/DS1")
        ds2_id = db.getObjectIdByPath("/DS2")

        value = db.getAttributeValue(ds1_id, "A1")
        self.assertEqual(value.shape, (2,))
        self.assertEqual(value.dtype.metadata.get("ref"), RegionReference)

        for raw in value:
            ref = RegionReference.frombytes(raw)
            self.assertEqual(ref.id, ds2_id)
            self.assertIsNone(ref.selection_bytes)
            self.assertEqual(ref.to_json(), {"id": getUuidFromId(ds2_id)})

        db.close()

    def testWriteReadRegionReferenceRoundTrip(self):
        filepath = "test/unit/out/h5py_test_testWriteReadRegionReferenceRoundTrip.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        wdb = Hdf5db(app_logger=self.log)
        wdb.writer = H5pyWriter(filepath, no_data=False)
        root_id = wdb.open()

        target_id = wdb.createDataset(shape=(6, 10), dtype=np.int32)
        wdb.createHardLink(root_id, "DS1", target_id)

        sel = selections.select((6, 10), (slice(0, 3), slice(2, 6)))
        ref = RegionReference("datasets/" + target_id, sel)
        dt = special_dtype(ref=RegionReference)
        wdb.createAttribute(root_id, "A1", np.array([ref.tobytes()], dtype=dt), dtype=dt)
        wdb.close()

        rdb = Hdf5db(app_logger=self.log)
        rdb.reader = H5pyReader(filepath, app_logger=self.log)
        root_id2 = rdb.open()
        target_id2 = rdb.getObjectIdByPath("/DS1")

        value = rdb.getAttributeValue(root_id2, "A1")
        read_ref = RegionReference.frombytes(value[0])
        # target dataset identity survives the round trip
        self.assertEqual(read_ref.id, target_id2)
        # the selection itself does not - only h5py's own low-level h5r/h5s
        # API can decode it, which this reader doesn't currently do
        self.assertIsNone(read_ref.selection_bytes)
        rdb.close()

    # --- H5pyWriter tests ---

    def testOpen(self):
        filepath = "test/unit/out/h5py_test_testOpen.h5"
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath)
        root_id = db.open()
        self.assertTrue(isSchema2Id(root_id))
        self.assertTrue(isRootObjId(root_id))
        self.assertFalse(db.closed)
        self.assertEqual(db.getObjectIdByPath("/"), root_id)
        db.close()
        self.assertTrue(db.closed)
        self.assertTrue(db.writer.isClosed())
        obj_id = db.open()
        self.assertEqual(obj_id, root_id)
        db.close()

    def testSimple(self):

        filepath = "test/unit/out/h5py_test_testSimple.h5"
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
        shape = (10, 10)
        dset_111_id = db.createDataset(shape=shape, dtype=np.int32)

        # try setting dset values with broadcasting
        arr_one_value = np.zeros((1, 1), dtype=np.int32)
        arr_one_value[0, 0] = 42
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_111_id, sel_all, arr_one_value)

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
            self.assertEqual(dset.shape, shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.assertEqual(dset[i, j], 42)
            self.assertTrue("g2" in f)
            g2 = f["g2"]
            self.assertTrue("extlink" in g2)
            self.assertTrue("slink" in g2)

        # write dataset values element by element
        db.open()
        arr = np.zeros(shape, dtype=np.int32)
        for i in range(shape[0]):
            for j in range(shape[1]):
                arr[i, j] = i * j
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_111_id, sel_all, arr)
        db.close()

        # verify changes in h5py
        with h5py.File(filepath) as f:
            dset = f["/g1/g1.1/dset1.1.1"]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.assertEqual(dset[i, j], i * j)

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
        sel = selections.select(shape, (slice(4, 5), slice(4, 5)))
        arr = np.zeros((1, 1), dtype=np.int32)
        arr[0, 0] = 42
        db.setDatasetValues(dset_111_id, sel, arr)
        db.close()

        with h5py.File(filepath) as f:
            dset = f["/g1/g1.1/dset1.1.1"]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if i == 4 and j == 4:
                        # this is the one element that was updated
                        expected = 42
                    else:
                        expected = i * j
                    self.assertEqual(dset[i, j], expected)

        # try a point write
        db.open()
        points = []
        for i in range(shape[0]):
            points.append((i, i))
        sel = selections.select(shape, points)
        arr = np.zeros((len(points),), dtype=np.int32)
        db.setDatasetValues(dset_111_id, sel, arr)
        db.close()

        with h5py.File(filepath) as f:
            dset = f["/g1/g1.1/dset1.1.1"]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if i == j:
                        # the diagonal elements were updated to 0
                        expected = 0
                    else:
                        expected = i * j
                    self.assertEqual(dset[i, j], expected)

        # try a fancy write (slice + coord list) — rows 0–2, columns [1, 3, 5]
        db.open()
        sel = selections.select(shape, (slice(0, 3), [1, 3, 5]))
        self.assertEqual(sel.select_type, selections.H5S_SEL_FANCY)
        arr = np.full((3, 3), 99, dtype=np.int32)
        db.setDatasetValues(dset_111_id, sel, arr)
        db.close()

        with h5py.File(filepath) as f:
            dset = f["/g1/g1.1/dset1.1.1"]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if i < 3 and j in (1, 3, 5):
                        expected = 99
                    elif i == j:
                        expected = 0  # zeroed by point write above
                    else:
                        expected = i * j
                    self.assertEqual(dset[i, j], expected)

    def testResizableDataset(self):
        filepath = "test/unit/out/h5py_test_testResizableDataset.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)

        nrows = 8
        ncols = 10
        shape = (nrows, ncols)
        dtype = np.int32
        maxdims = (None, ncols * 2)
        layout = {"class": "H5D_CHUNKED", "dims": (nrows, ncols)}
        cpl = {"layout": layout}

        root_id = db.open()
        dset_id = db.createDataset(shape, maxdims=maxdims, dtype=dtype, cpl=cpl)
        db.createHardLink(root_id, "dset", dset_id)
        db.createAttribute(dset_id, "a1", "Hello, world")

        # resize limited dimension
        db.resizeDataset(dset_id, (nrows, ncols * 2))

        # try to go beyond max extent
        try:
            db.resizeDataset(dset_id, (nrows, ncols * 3))
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        db.close()

        with h5py.File(filepath) as f:
            dset = f["dset"]
            self.assertEqual(dset.shape, (nrows, ncols * 2))

        db.open()
        # resize unlimited dimension
        db.resizeDataset(dset_id, (nrows * 10, ncols))

        db.close()

        with h5py.File(filepath) as f:
            dset = f["dset"]
            self.assertEqual(dset.shape, (nrows * 10, ncols))

    def testNullSpaceAttribute(self):

        filepath = "test/unit/out/h5py_test_testNullSpaceAttribute.h5"
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

        filepath = "test/unit/out/h5py_test_testNullScalarAttribute.h5"
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

        filepath = "test/unit/out/h5py_test_testFixedStringAttribute.h5"
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

        filepath = "test/unit/out/h5py_test_testVlenAsciiAttribute.h5"
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

        filepath = "test/unit/out/h5py_test_testVlenUtf8Attribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        value = "one: 一"

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

        filepath = "test/unit/out/h5py_test_testIntAttribute.h5"
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

        filepath = "test/unit/out/h5py_test_testCreateReferenceAttribute.h5"
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

        filepath = "test/unit/out/h5py_test_testVlenReferenceAttribute.h5"
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

    def testCreateRegionReferencePointsAttribute(self):
        filepath = "test/unit/out/h5py_test_testCreateRegionReferencePointsAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()

        target_id = db.createDataset(shape=(3, 16), dtype=np.int32)
        db.createHardLink(root_id, "DS1", target_id)

        sel = selections.select((3, 16), ([0, 2, 1, 2], [1, 11, 0, 4]))
        ref = RegionReference("datasets/" + target_id, sel)
        dt = special_dtype(ref=RegionReference)
        db.createAttribute(root_id, "A1", np.array([ref.tobytes()], dtype=dt), dtype=dt)
        db.close()

        with h5py.File(filepath) as f:
            self.assertTrue("A1" in f.attrs)
            a1 = f.attrs["A1"]
            hdf5_ref = a1[0]
            self.assertTrue(hdf5_ref)  # not a null reference
            sid = h5r.get_region(hdf5_ref, f["DS1"].id)
            self.assertEqual(sid.get_select_type(), selections.H5S_SEL_POINTS)
            self.assertEqual(sid.get_select_npoints(), 4)
            points = sid.get_select_elem_pointlist().tolist()
            self.assertEqual(points, [[0, 1], [2, 11], [1, 0], [2, 4]])

    def testCreateRegionReferenceHyperslabAttribute(self):
        filepath = "test/unit/out/h5py_test_testCreateRegionReferenceHyperslabAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()

        target_id = db.createDataset(shape=(6, 10), dtype=np.int32)
        db.createHardLink(root_id, "DS1", target_id)

        sel = selections.select((6, 10), (slice(0, 3), slice(2, 6)))
        ref = RegionReference("datasets/" + target_id, sel)
        dt = special_dtype(ref=RegionReference)
        db.createAttribute(root_id, "A1", np.array([ref.tobytes()], dtype=dt), dtype=dt)
        db.close()

        with h5py.File(filepath) as f:
            hdf5_ref = f.attrs["A1"][0]
            self.assertTrue(hdf5_ref)
            sid = h5r.get_region(hdf5_ref, f["DS1"].id)
            self.assertEqual(sid.get_select_type(), selections.H5S_SEL_HYPERSLABS)
            self.assertEqual(sid.get_select_npoints(), 3 * 4)
            self.assertEqual(sid.get_select_bounds(), ((0, 2), (2, 5)))

    def testCreateRegionReferenceNullAttribute(self):
        filepath = "test/unit/out/h5py_test_testCreateRegionReferenceNullAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()

        dt = special_dtype(ref=RegionReference)
        db.createAttribute(root_id, "A1", np.array([b''], dtype=dt), dtype=dt)
        db.close()

        with h5py.File(filepath) as f:
            hdf5_ref = f.attrs["A1"][0]
            self.assertFalse(hdf5_ref)  # null reference

    def testCreateRegionReferenceFancyAttribute(self):
        # H5S_SEL_FANCY (mixed slice + coordinate list) has no equivalent in
        # a real HDF5 region reference's point/hyperslab representation, so
        # RegionReference.to_json() falls back to embedding the fully
        # general Selection.to_dict() under a "selection_dict" key (see
        # hdf5dtype.py).  Confirm that lets it flow all the way through
        # Hdf5db.createAttribute() and be written as a real HDF5 region
        # reference (decomposed into a union of hyperslabs, same as HDF5
        # itself would do for e.g. dset[0:4, [1, 3, 7]]).
        filepath = "test/unit/out/h5py_test_testCreateRegionReferenceFancyAttribute.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()

        target_id = db.createDataset(shape=(6, 10), dtype=np.int32)
        db.createHardLink(root_id, "DS1", target_id)

        sel = selections.select((6, 10), (slice(0, 4), [1, 3, 7]))
        ref = RegionReference("datasets/" + target_id, sel)
        d = ref.to_json()
        self.assertIn("selection_dict", d)

        dt = special_dtype(ref=RegionReference)
        db.createAttribute(root_id, "A1", np.array([ref.tobytes()], dtype=dt), dtype=dt)
        attr = db.getAttribute(root_id, "A1")
        self.assertIn("selection_dict", attr["value"][0])
        db.close()

        with h5py.File(filepath) as f:
            hdf5_ref = f.attrs["A1"][0]
            self.assertTrue(hdf5_ref)  # not a null reference
            sid = h5r.get_region(hdf5_ref, f["DS1"].id)
            self.assertEqual(sid.get_select_type(), selections.H5S_SEL_HYPERSLABS)
            self.assertEqual(sid.get_select_npoints(), 4 * 3)  # 4 rows x 3 fancy columns

    def testBuildRegionDataspaceFancySelection(self):
        # Exercise H5pyWriter._buildRegionDataspace() directly against a
        # FANCY selection, independent of the JSON round trip covered above.
        filepath = "test/unit/out/h5py_test_testBuildRegionDataspaceFancySelection.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        writer = H5pyWriter(filepath, no_data=False)
        with h5py.File(filepath, "w") as f:
            target = f.create_dataset("DS1", data=np.zeros((6, 10), dtype=np.int32))

            sel = selections.select((6, 10), (slice(0, 4), [1, 3, 7]))
            ref = RegionReference()
            ref._selection_bytes = sel.tobytes()  # _buildRegionDataspace only needs the selection

            sid = writer._buildRegionDataspace(target, ref)
            self.assertEqual(sid.get_select_type(), selections.H5S_SEL_HYPERSLABS)
            self.assertEqual(sid.get_select_npoints(), 4 * 3)  # 4 rows x 3 fancy columns

            hdf5_ref = h5r.create(target.id, b'.', h5r.DATASET_REGION, sid)
            sid2 = h5r.get_region(hdf5_ref, target.id)
            self.assertEqual(sid2.get_select_npoints(), 12)

    def testVlenStringDataset(self):
        filepath = "test/unit/out/h5py_test_testVlenStringDataset.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run
        nrows = 4
        shape = (nrows,)
        dtype = special_dtype(vlen=str)
        data = ["Hello", "HDF5", "REST", "API"]
        init_arr = np.array(data, dtype=dtype)

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)

        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        db.createHardLink(root_id, "dset", dset_id)
        sel_all = selections.select(shape, ...)
        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, shape)

        db.setDatasetValues(dset_id, sel_all, init_arr)

        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertTrue(np.array_equal(arr, init_arr))
        sel_one = selections.select(shape, slice(2, 3))
        arr = db.getDatasetValues(dset_id, sel_one)
        self.assertEqual(arr.shape, (1,))
        self.assertEqual(arr[0], 'REST')

        db.close()

        with h5py.File(filepath) as f:
            self.assertTrue("dset" in f)
            dset = f["dset"]
            self.assertEqual(dset.shape, (nrows,))
            self.assertEqual(dset.dtype, dtype)
            for i in range(nrows):
                self.assertEqual(dset[i], data[i].encode())

    def testCommittedType(self):

        filepath = "test/unit/out/h5py_test_testCommittedType.h5"
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

        filepath = "test/unit/out/h5py_test_testCommittedCompoundType.h5"
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
        file_out = "test/unit/out/h5py_test_testReaderWithUpdate.h5"
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
        arr = arr.reshape(1, 10)
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

        filepath = "test/unit/out/h5py_test_testCompression.h5"
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

    def testQueryDatasetChunked(self):
        # verifies queryDataset works correctly both before a flush (querying
        # in-memory updates directly) and after (querying persisted, chunked
        # data through H5pyReader via Hdf5db.getChunkIterator)
        filepath = "test/unit/out/h5py_test_testQueryDatasetChunked.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        dtype = np.dtype([("symbol", "S4"), ("date", "S8"), ("open", "i4"), ("close", "i4")])
        rows = [
            ("EBAY", "20170102", 3023, 3088),
            ("AAPL", "20170102", 3054, 2933),
            ("AMZN", "20170102", 2973, 3011),
            ("EBAY", "20170103", 3042, 3128),
            ("AAPL", "20170103", 3182, 3034),
            ("AMZN", "20170103", 3021, 2788),
            ("EBAY", "20170104", 2798, 2876),
            ("AAPL", "20170104", 2834, 2867),
            ("AMZN", "20170104", 2891, 2978),
            ("EBAY", "20170105", 2973, 2962),
            ("AAPL", "20170105", 2934, 3010),
            ("AMZN", "20170105", 3018, 3086),
        ]
        shape = (len(rows),)
        arr = np.zeros(shape, dtype=dtype)
        for i, row in enumerate(rows):
            for j in range(4):
                arr[i][j] = row[j]

        # small chunk size so the AAPL matches (indices 1, 4, 7, 10) each land
        # in a different chunk once the dataset is read back through the reader
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (3,)}}
        query = "symbol == b'AAPL'"
        expected_indexes = {1, 4, 7, 10}

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype, cpl=cpl)
        db.createHardLink(root_id, "trades", dset_id)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)

        # query before flush - values only exist as an in-memory update
        indices = db.queryDataset(dset_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (4, 1))
        for idx in indices:
            self.assertIn(int(idx[0]), expected_indexes)

        db.close()  # flushes the dataset to storage

        # reopen with a real reader - queryDataset must now read the
        # persisted, chunked data from storage
        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(filepath, app_logger=self.log)
        db.open()
        dset_id = db.getObjectIdByPath("/trades")

        indices = db.queryDataset(dset_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (4, 1))
        for idx in indices:
            self.assertIn(int(idx[0]), expected_indexes)

        # limit should stop early while preserving ascending order
        limited = db.queryDataset(dset_id, query, limit=2)
        self.assertEqual(limited.shape, (2, 1))
        self.assertEqual([int(x[0]) for x in limited], [1, 4])

        # selection restricted to rows 2-11 spans multiple chunks too
        sel = selections.select(shape, slice(2, 12))
        indices = db.queryDataset(dset_id, query, sel=sel)
        self.assertEqual(indices.shape, (3, 1))
        expected_in_order = (4, 7, 10)
        for i, idx in enumerate(indices):
            self.assertEqual(int(idx[0]), expected_in_order[i])

        db.close()

    def testGetDatasetValuesByQueryChunked(self):
        # verifies Hdf5db.getDatasetValues(..., query=...) works correctly both
        # before a flush (filtering in-memory updates directly) and after
        # (filtering persisted, chunked data read through H5pyReader via
        # Hdf5db.getChunkIterator)
        filepath = "test/unit/out/h5py_test_testGetDatasetValuesByQueryChunked.h5"
        if os.path.isfile(filepath):
            os.remove(filepath)  # cleanup any previous run

        dtype = np.dtype([("symbol", "S4"), ("date", "S8"), ("open", "i4"), ("close", "i4")])
        rows = [
            ("EBAY", "20170102", 3023, 3088),
            ("AAPL", "20170102", 3054, 2933),
            ("AMZN", "20170102", 2973, 3011),
            ("EBAY", "20170103", 3042, 3128),
            ("AAPL", "20170103", 3182, 3034),
            ("AMZN", "20170103", 3021, 2788),
            ("EBAY", "20170104", 2798, 2876),
            ("AAPL", "20170104", 2834, 2867),
            ("AMZN", "20170104", 2891, 2978),
            ("EBAY", "20170105", 2973, 2962),
            ("AAPL", "20170105", 2934, 3010),
            ("AMZN", "20170105", 3018, 3086),
        ]
        shape = (len(rows),)
        arr = np.zeros(shape, dtype=dtype)
        for i, row in enumerate(rows):
            for j in range(4):
                arr[i][j] = row[j]

        # small chunk size so the AAPL matches (indices 1, 4, 7, 10) each land
        # in a different chunk once the dataset is read back through the reader
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (3,)}}
        query = "symbol == b'AAPL'"
        expected_indexes = (1, 4, 7, 10)

        db = Hdf5db(app_logger=self.log)
        db.writer = H5pyWriter(filepath, no_data=False)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype, cpl=cpl)
        db.createHardLink(root_id, "trades", dset_id)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)

        # query before flush - values only exist as an in-memory update
        values = db.getDatasetValues(dset_id, sel_all, query=query)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.shape, (4,))
        for i, val in enumerate(values):
            self.assertEqual(val, arr[expected_indexes[i]])

        db.close()  # flushes the dataset to storage

        # reopen with a real reader - getDatasetValues must now filter the
        # persisted, chunked data read from storage rather than in-memory updates
        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(filepath, app_logger=self.log)
        db.open()
        dset_id = db.getObjectIdByPath("/trades")

        values = db.getDatasetValues(dset_id, sel_all, query=query)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.dtype, dtype)
        self.assertEqual(values.shape, (4,))
        for i, val in enumerate(values):
            self.assertEqual(val, arr[expected_indexes[i]])

        # selection restricted to rows 2-11 spans multiple chunks too
        sel = selections.select(shape, slice(2, 12))
        values = db.getDatasetValues(dset_id, sel, query=query)
        self.assertEqual(values.shape, (3,))
        for i, val in enumerate(values):
            self.assertEqual(val, arr[expected_indexes[i + 1]])

        # a query with no matches returns an empty, correctly-typed array
        empty = db.getDatasetValues(dset_id, sel_all, query="symbol == b'XYZ'")
        self.assertEqual(empty.dtype, dtype)
        self.assertEqual(empty.shape, (0,))

        db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
