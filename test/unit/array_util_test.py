##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HSDS (HDF5 Scalable Data Service), Libraries and      #
# Utilities.  The full HSDS copyright notice, including                      #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
import unittest
import json
import numpy as np

import base64

from h5json.array_util import bytesArrayToList
from h5json.array_util import toTuple
from h5json.array_util import getNumElements
from h5json.array_util import jsonToArray
from h5json.array_util import arrayToBytes
from h5json.array_util import bytesToArray
from h5json.array_util import getByteArraySize
from h5json.array_util import IndexIterator
from h5json.array_util import ndarray_compare
from h5json.array_util import getNumpyValue
from h5json.array_util import getBroadcastShape
from h5json.array_util import isVlen

from h5json.hdf5dtype import special_dtype
from h5json.hdf5dtype import check_dtype
from h5json.hdf5dtype import createDataType
from h5json.hdf5dtype import RegionReference
from h5json.objid import createObjId, getUuidFromId
from h5json import selections


class ArrayUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ArrayUtilTest, self).__init__(*args, **kwargs)
        # main

    def testByteArrayToList(self):
        data_items = (
            42,
            "foo",
            b"foo",
            [1, 2, 3],
            (1, 2, 3),
            ["A", "B", "C"],
            [b"A", b"B", b"C"],
            [["A", "B"], [b"a", b"b", b"c"]],
        )
        for data in data_items:
            json_data = bytesArrayToList(data)
            # will throw TypeError if not able to convert
            json.dumps(json_data)

    def testByteArrayToListRegionReference(self):
        # matches the format used in data/json/regionref_dset.json /
        # regionref_attr.json: {"id": <bare uuid>, "select_type": ..., "selection": [...]}
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)

        pts_sel = selections.select((3, 16), ([0, 2, 1, 2], [1, 11, 0, 4]))
        ref_pts = RegionReference(dset_id, pts_sel)

        hs_sel = selections.select((3, 16), (slice(0, 2), slice(0, 4)))
        ref_hs = RegionReference(dset_id, hs_sel)

        dt = special_dtype(ref=RegionReference)

        # 1-D array of two region refs
        arr = np.empty((2,), dtype=dt)
        arr[0] = ref_pts.tobytes()
        arr[1] = ref_hs.tobytes()
        result = bytesArrayToList(arr)
        json.dumps(result)  # must be JSON-serializable
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], getUuidFromId(dset_id))
        self.assertEqual(result[0]["select_type"], "H5S_SEL_POINTS")
        self.assertEqual(result[0]["selection"], [[0, 1], [2, 11], [1, 0], [2, 4]])
        self.assertEqual(result[1]["select_type"], "H5S_SEL_HYPERSLABS")
        self.assertEqual(result[1]["selection"], [[[0, 0], [1, 3]]])

        # scalar (0-d) region ref array
        arr0 = np.empty((), dtype=dt)
        arr0[()] = ref_pts.tobytes()
        result0 = bytesArrayToList(arr0)
        self.assertEqual(result0["select_type"], "H5S_SEL_POINTS")

        # 2-D array of region refs
        arr2 = np.empty((2, 1), dtype=dt)
        arr2[0, 0] = ref_pts.tobytes()
        arr2[1, 0] = ref_hs.tobytes()
        result2 = bytesArrayToList(arr2)
        json.dumps(result2)
        self.assertEqual(result2[0][0]["select_type"], "H5S_SEL_POINTS")
        self.assertEqual(result2[1][0]["select_type"], "H5S_SEL_HYPERSLABS")

        # unset (null) region ref -> None
        arrn = np.empty((1,), dtype=dt)
        arrn[0] = b''
        self.assertEqual(bytesArrayToList(arrn), [None])

    def testJsonToArrayRegionReference(self):
        # inverse of testByteArrayToListRegionReference() - and matches the
        # format used in data/json/regionref_dset.json / regionref_attr.json
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)
        dt = special_dtype(ref=RegionReference)

        points_json = {
            "id": getUuidFromId(dset_id),
            "select_type": "H5S_SEL_POINTS",
            "selection": [[0, 1], [2, 11], [1, 0], [2, 4]],
        }
        hyperslab_json = {
            "id": getUuidFromId(dset_id),
            "select_type": "H5S_SEL_HYPERSLABS",
            "selection": [[[0, 0], [1, 3]]],
        }

        # 1-D array of two region refs, plus an unset (None) element
        arr = jsonToArray((3,), dt, [points_json, hyperslab_json, None])
        self.assertEqual(arr.shape, (3,))
        self.assertEqual(arr.dtype, dt)

        ref0 = RegionReference.frombytes(arr[0])
        self.assertEqual(ref0.to_json(), points_json)
        ref1 = RegionReference.frombytes(arr[1])
        self.assertEqual(ref1.to_json(), hyperslab_json)
        self.assertEqual(arr[2], b'')

        # scalar (0-d)
        arr0 = jsonToArray((), dt, points_json)
        self.assertEqual(arr0.shape, ())
        self.assertEqual(RegionReference.frombytes(arr0[()]).to_json(), points_json)

        # scalar attribute case: hsds represents an H5S_SCALAR shape as
        # np_dims=[1] (rather than ()) without wrapping the JSON value in a
        # matching 1-element list - a bare dict (or None) must still be
        # treated as the single leaf value, not iterated over as if it were
        # a list of per-element values (regression test for a bug where
        # enumerate() over the dict's keys was passed to RegionReference.from_json())
        arr1 = jsonToArray((1,), dt, points_json)
        self.assertEqual(arr1.shape, (1,))
        self.assertEqual(RegionReference.frombytes(arr1[0]).to_json(), points_json)

        # round trip through bytesArrayToList() and back
        as_list = bytesArrayToList(arr)
        arr_rt = jsonToArray((3,), dt, as_list)
        self.assertEqual(RegionReference.frombytes(arr_rt[0]).to_json(), points_json)
        self.assertEqual(RegionReference.frombytes(arr_rt[1]).to_json(), hyperslab_json)
        self.assertEqual(arr_rt[2], b'')

    def testJsonToArrayRegionReferenceFancySelection(self):
        # a FANCY selection has no points/hyperslab equivalent, so its JSON
        # form uses a "selection_dict" key (see RegionReference.to_json())
        # instead of "select_type"/"selection" - confirm it still round-trips
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)
        dt = special_dtype(ref=RegionReference)

        sel = selections.select((6, 10), (slice(0, 4), [1, 3, 7]))
        ref = RegionReference("datasets/" + dset_id, sel)
        fancy_json = ref.to_json()
        self.assertIn("selection_dict", fancy_json)

        arr = jsonToArray((1,), dt, [fancy_json])
        ref2 = RegionReference.frombytes(arr[0])
        self.assertEqual(ref2.to_json(), fancy_json)
        sel2 = selections.Selection.frombytes(ref2.selection_bytes)
        self.assertEqual(sel2, sel)

        as_list = bytesArrayToList(arr)
        self.assertEqual(as_list, [fancy_json])

    def testToTuple(self):
        data0d = 42  # scalar
        data1d1 = [1]  # one dimensional, one element list
        data1d = [1, 2, 3, 4, 5]  # list
        data2d1 = [
            [1, 2],
        ]  # two dimensional, one element
        data2d = [[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]]  # list of two-element lists
        data3d = [[[0, 0.0], [1, 0.1]], [[2, 0.2], [3, 0.3]]]  # list of list of lists
        out = toTuple(0, data0d)
        self.assertEqual(data0d, out)
        out = toTuple(1, data1d1)
        self.assertEqual(data1d1, out)
        out = toTuple(1, data1d)
        self.assertEqual(data1d, out)
        out = toTuple(2, data2d)
        self.assertEqual(data2d, out)
        out = toTuple(1, data2d1)
        self.assertEqual([(1, 2)], out)
        out = toTuple(3, data3d)
        self.assertEqual(data3d, out)
        out = toTuple(1, data2d)  # treat input as 1d array of two-field compound types
        self.assertEqual([(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)], out)
        out = toTuple(2, data3d)  # treat input as 2d array of two-field compound types
        self.assertEqual([[(0, 0.0), (1, 0.1)], [(2, 0.2), (3, 0.3)]], out)
        out = toTuple(1, data3d)  # treat input a 1d array of compound type of compound types
        self.assertEqual([((0, 0.0), (1, 0.1)), ((2, 0.2), (3, 0.3))], out)

    def testToTupleStrData(self):
        data = "a string!"
        out = toTuple(0, data)
        self.assertEqual(data, out)

        data = ["a string!"]
        out = toTuple(1, data)
        self.assertEqual(data, out)

        data = ["a string2"]
        out = toTuple(1, data)
        self.assertEqual(data, out)

        data = [["partA", "partB", "partC"],]
        out = toTuple(1, data)
        self.assertEqual([("partA", "partB", "partC"), ], out)

        data = [[[4, 8, 12], "four"], [[5, 10, 15], "five"]]
        out = toTuple(1, data)
        self.assertEqual([((4, 8, 12), 'four'), ((5, 10, 15), 'five')], out)

    def testGetNumElements(self):
        shape = (4,)
        nelements = getNumElements(shape)
        self.assertEqual(nelements, 4)

        shape = [10,]
        nelements = getNumElements(shape)
        self.assertEqual(nelements, 10)

        shape = (10, 8)
        nelements = getNumElements(shape)
        self.assertEqual(nelements, 80)

    def testJsonToArray(self):

        # simple integer
        dt = np.dtype("i4")
        shape = [4, ]
        data = [0, 2, 4, 6]
        out = jsonToArray(shape, dt, data)

        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (4,))
        for i in range(4):
            self.assertEqual(out[i], i * 2)

        shape = ()  # scalar
        data = 42
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, ())
        self.assertEqual(out[()], 42)

        shape = (1, )  # one element
        data = 42
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (1, ))
        self.assertEqual(out[0], 42)

        shape = (10, )  # multi-1D
        data = list(range(10))
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (10, ))
        self.assertEqual(out[5], 5)

        shape = (5, 4)  # multi-2D
        data = []
        for i in range(5):
            data.append([42, ] * 4)
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (5, 4))
        self.assertEqual(out[2, 3], 42)

        shape = (5, 4)  # multi-2D, reshape input data
        data = [42, ] * 20
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (5, 4))
        self.assertEqual(out[2, 3], 42)

        dt = np.dtype("S10")  # fixed size string
        shape = [5, ]
        data = ["parting", "is", "such", "sweet", "sorrow"]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (5, ))
        self.assertEqual(out[4], b'sorrow')

        shape = ()  # scalar
        data = "a string"
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, ())
        self.assertEqual(out[()], b'a string')

        # VLEN Scalar str
        dt = special_dtype(vlen=str)
        data = "I'm a string!"
        shape = []
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, ())
        val = out[()]
        self.assertEqual(val, data)

        # VLEN one element str
        dt = special_dtype(vlen=str)
        data = "I'm a string!"
        shape = [1,]
        out = jsonToArray(shape, dt, [data,])
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (1,))
        val = out[0]
        self.assertEqual(val, data)

        # VLEN multi element
        shape = [5, ]
        data = ["parting", "is", "such", "sweet", "sorrow"]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (5, ))
        self.assertEqual(out[4], 'sorrow')

        # VLEN ascii
        dt = special_dtype(vlen=bytes)
        data = [b"one", b"two", b"three", b"four", b"five"]
        shape = [5, ]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (5,))
        self.assertTrue("vlen" in out.dtype.metadata)
        self.assertEqual(out.dtype.metadata["vlen"], bytes)
        self.assertEqual(out.dtype.kind, "O")
        self.assertEqual(out.shape, (5,))
        # TBD: code does not actually enforce use of bytes vs. str,
        #  probably not worth the effort to fix
        self.assertEqual(out[2], b"three")
        self.assertEqual(out[3], b"four")

        # VLEN unicode
        dt = special_dtype(vlen=bytes)
        data = ["one", "two", "three", "four", "five"]
        shape = [5, ]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertTrue("vlen" in out.dtype.metadata)
        self.assertEqual(out.dtype.metadata["vlen"], bytes)
        self.assertEqual(out.dtype.kind, "O")
        e = out[2]
        self.assertEqual(e, "three")

        # test utf8 strings
        dt = np.dtype("S26")
        shape = []
        data = "eight: \u516b"
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out[()], data.encode())

        dt = special_dtype(vlen=str)
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out[()], data)

        data = ["I'm an UTF-8 null terminated string",]
        shape = [1,]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out[0], data[0])

        dt = np.dtype("S12")
        data = "eight: \u516b"
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out[()], data.encode("utf8"))

        # UTF8 encode the data first
        out = jsonToArray(shape, dt, data.encode('utf8'))
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out[()], data.encode('utf8'))

        # one-element array
        shape = [1,]
        dt = np.dtype("S12")
        data = "eight: \u516b"
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out[0], b'eight: \xe5\x85\xab')

        # VLEN data
        shape = []
        dt = special_dtype(vlen=np.dtype("S10"))
        data = [("foo", "bar")]
        out = jsonToArray(shape, dt, data)

        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, ())
        self.assertEqual(out[()][0], b'foo')
        self.assertEqual(out[()][1], b'bar')

        dt = special_dtype(vlen=np.dtype("int32"))
        shape = [4, ]
        data = [
            [1,],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
        ]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(check_dtype(vlen=out.dtype), np.dtype("int32"))

        self.assertEqual(out.shape, (4,))
        self.assertEqual(out.dtype.kind, "O")
        self.assertEqual(check_dtype(vlen=out.dtype), np.dtype("int32"))
        for i in range(4):
            e = out[i]  # .tolist()
            self.assertTrue(isinstance(e, np.ndarray))
            self.assertEqual(e.shape, (i + 1,))
            self.assertEqual(e.dtype, np.dtype("int32"))
            for j in range(i + 1):
                self.assertEqual(e[j], j + 1)

        # VLEN 2D data
        dt = special_dtype(vlen=np.dtype("int32"))
        shape = [2, 2]
        data = [
            [
                [0,],
                [1, 2],
            ],
            [
                [1,],
                [2, 3],
            ],
        ]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(check_dtype(vlen=out.dtype), np.dtype("int32"))

        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out.dtype.kind, "O")
        self.assertEqual(check_dtype(vlen=out.dtype), np.dtype("int32"))
        e = out[0, 0]
        self.assertTrue(isinstance(e, np.ndarray))
        self.assertEqual(list(e), [0])
        e = out[0, 1]
        self.assertTrue(isinstance(e, np.ndarray))
        self.assertEqual(list(e), [1, 2])
        e = out[1, 0]
        self.assertTrue(isinstance(e, np.ndarray))
        self.assertEqual(list(e), [1])
        e = out[1, 1]
        self.assertTrue(isinstance(e, np.ndarray))
        self.assertEqual(list(e), [2, 3])

        # create VLEN of obj ref's
        ref_type = {"class": "H5T_REFERENCE", "base": "H5T_STD_REF_OBJ"}
        vlen_type = {"class": "H5T_VLEN", "base": ref_type}
        dt = createDataType(vlen_type)  # np datatype

        id0 = b"g-a4f455b2-c8cf-11e7-8b73-0242ac110009"
        id1 = b"g-a50af844-c8cf-11e7-8b73-0242ac110009"
        id2 = b"g-a5236276-c8cf-11e7-8b73-0242ac110009"

        data = [
            [id0, ],
            [id0, id1],
            [id0, id1, id2],
        ]
        shape = [3, ]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        base_type = check_dtype(vlen=out.dtype)
        self.assertEqual(base_type.kind, "S")
        self.assertEqual(base_type.itemsize, 48)

        self.assertEqual(out.shape, (3,))
        self.assertEqual(out.dtype.kind, "O")
        self.assertEqual(check_dtype(vlen=out.dtype), np.dtype("S48"))

        e = out[0]
        self.assertTrue(isinstance(e, np.ndarray))
        self.assertEqual(list(e), [id0,])
        e = out[1]
        self.assertTrue(isinstance(e, np.ndarray))
        self.assertEqual(list(e), [id0, id1])
        e = out[2]
        self.assertTrue(isinstance(e, np.ndarray))
        self.assertEqual(list(e), [id0, id1, id2])

        # compound type
        dt = np.dtype([("a", "i4"), ("b", "S5")])
        shape = [2, ]
        data = [[4, "four"], [5, "five"]]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))

        self.assertEqual(out.shape, (2,))
        self.assertTrue(isinstance(out[0], np.void))
        e0 = out[0].tolist()
        self.assertEqual(e0, (4, b"four"))
        self.assertTrue(isinstance(out[1], np.void))
        e1 = out[1].tolist()
        self.assertEqual(e1, (5, b"five"))

        data = [[6, "six"],]
        shape = [1,]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (1,))
        self.assertTrue(isinstance(out[0], np.void))
        e1 = out[0].tolist()
        self.assertEqual(e1, (6, b"six"))

        data = [7, "seven"]
        shape = []
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, ())
        self.assertTrue(isinstance(out[()], np.void))
        e1 = out[()].tolist()
        self.assertEqual(e1, (7, b"seven"))

        data = [8, "eight"]
        shape = [1,]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (1,))
        self.assertTrue(isinstance(out[0], np.void))
        e1 = out[0].tolist()
        self.assertEqual(e1, (8, b"eight"))

        dt = np.dtype([("a", "i4"), ("b", "f4")])
        shape = [1, ]
        data = [42, 0.42]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (1, ))
        e1 = out[0]
        self.assertEqual(e1[0], 42)

        # compound with VLEN element

        dt_str = special_dtype(vlen=str)
        dt = np.dtype([("a", "i4"), ("b", dt_str)])
        shape = [2, ]
        data = [[4, "four"], [5, "five"]]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (2,))
        e0 = out[0].tolist()
        self.assertEqual(e0, (4, "four"))

        shape = [1, ]
        data = [[6, "six"],]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (1,))
        e0 = out[0].tolist()
        self.assertEqual(e0, (6, "six"))

        shape = []
        data = [7, "seven",]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, ())
        e0 = out[()]
        self.assertEqual(len(e0), 2)
        self.assertEqual(e0[0], 7)
        self.assertEqual(e0[1], "seven")

        # compound type with array field
        dt = np.dtype([("a", ("i4", 3)), ("b", "S5")])
        shape = [2, ]
        data = [[[4, 8, 12], "four"], [[5, 10, 15], "five"]]
        out = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(out, np.ndarray))

        self.assertEqual(out.shape, (2,))
        self.assertTrue(isinstance(out[0], np.void))
        e0 = out[0]
        self.assertEqual(len(e0), 2)
        e0a = e0[0]
        self.assertTrue(isinstance(e0a, np.ndarray))
        self.assertEqual(e0a[0], 4)
        self.assertEqual(e0a[1], 8)
        self.assertEqual(e0a[2], 12)
        e0b = e0[1]
        self.assertEqual(e0b, b"four")
        self.assertTrue(isinstance(out[1], np.void))
        e1 = out[1]
        self.assertEqual(len(e1), 2)
        e1a = e1[0]
        self.assertTrue(isinstance(e1a, np.ndarray))
        self.assertEqual(e1a[0], 5)
        self.assertEqual(e1a[1], 10)
        self.assertEqual(e1a[2], 15)
        e1b = e1[1]
        self.assertEqual(e1b, b"five")

    def testToBytes(self):
        # Simple array
        dt = np.dtype("<i4")
        arr = np.asarray((1, 2, 3, 4), dtype=dt)
        buffer = arrayToBytes(arr)
        self.assertEqual(buffer, arr.tobytes())

        # convert buffer back to arr
        arr_copy = bytesToArray(buffer, dt, (4,))
        self.assertTrue(np.array_equal(arr, arr_copy))

        # big-endian ints
        dt = np.dtype(">u8")
        arr = np.asarray((1, 2, 3, 4), dtype=dt)
        buffer = arrayToBytes(arr)
        self.assertEqual(buffer, arr.tobytes())

        # fixed length string
        dt = np.dtype("S8")
        arr = np.asarray(("abcdefgh", "ABCDEFGH", "12345678"), dtype=dt)
        buffer = arrayToBytes(arr)
        self.assertEqual(buffer, arr.tobytes())

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (3,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # fixed length UTF8 string
        dt = np.dtype("S10")
        arr = np.asarray(b'eight: \xe5\x85\xab', dtype=dt)
        buffer = arrayToBytes(arr)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, ())
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # invalid UTF string
        dt = np.dtype("S2")
        arr = np.asarray(b'\xff\xfe', dtype=dt)
        buffer = arrayToBytes(arr)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, ())
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # invalid UTF string with base64 encoding
        dt = np.dtype("S2")
        arr = np.asarray(b'\xff\xfe', dtype=dt)
        buffer = b'//4='  # this is the base64 encoding of b'\xff\xfe'

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # Compound non-vlen
        dt = np.dtype([("x", "f8"), ("y", "i4")])
        arr = np.zeros((4,), dtype=dt)
        arr[0] = (3.12, 42)
        arr[3] = (1.28, 69)
        buffer = arrayToBytes(arr)
        self.assertEqual(buffer, arr.tobytes())

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # VLEN of int32's
        dt = special_dtype(vlen=np.dtype("<i4"))
        arr = np.zeros((4,), dtype=dt)
        arr[0] = np.int32([1, ])
        arr[1] = np.int32([1, 2])
        arr[2] = 0  # test un-initialized value
        arr[3] = np.int32([1, 2, 3])
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 40)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # VLEN of generic object ndarray
        arr = np.zeros((4,), dtype=object)

        try:
            arrayToBytes(arr)
            self.assertTrue(False)  # expected type error
        except TypeError:
            pass  # expected, object arrays not supported for arrayToBytes

        # RegionReference: also an object ("O") dtype, but tagged with 'ref'
        # metadata instead of 'vlen' - unlike the plain object array above,
        # this must succeed: each element is variable-length raw bytes
        # (RegionReference.tobytes()) and should serialize the same
        # length-prefixed way vlen bytes/str elements do.
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)
        dt = special_dtype(ref=RegionReference)

        pts_sel = selections.select((3, 16), ([0, 2], [1, 11]))
        ref_pts = RegionReference(dset_id, pts_sel)
        hs_sel = selections.select((3, 16), (slice(0, 2), slice(0, 4)))
        ref_hs = RegionReference(dset_id, hs_sel)

        arr = np.zeros((3,), dtype=dt)
        arr[0] = ref_pts.tobytes()
        arr[1] = ref_hs.tobytes()
        arr[2] = 0  # uninitialized/never-written element (as np.zeros leaves it)

        count = getByteArraySize(arr)
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), count)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (3,))
        self.assertEqual(arr_copy[0], ref_pts.tobytes())
        self.assertEqual(arr_copy[1], ref_hs.tobytes())
        # the uninitialized element round-trips to an empty (null) ref
        self.assertEqual(arr_copy[2], b'')

        as_list = bytesArrayToList(arr_copy)
        self.assertEqual(as_list[0]["id"], getUuidFromId(dset_id))
        self.assertEqual(as_list[0]["select_type"], "H5S_SEL_POINTS")
        self.assertEqual(as_list[1]["select_type"], "H5S_SEL_HYPERSLABS")
        self.assertIsNone(as_list[2])

        # an explicit null ref (b'') also round-trips correctly
        arr_null = np.zeros((1,), dtype=dt)
        arr_null[0] = b''
        buffer_null = arrayToBytes(arr_null)
        arr_null_copy = bytesToArray(buffer_null, dt, (1,))
        self.assertEqual(arr_null_copy[0], b'')

        # VLEN of strings
        dt = special_dtype(vlen=str)
        arr = np.zeros((5,), dtype=dt)
        arr[0] = "one: \u4e00"
        arr[1] = "two: \u4e8c"
        arr[2] = "three: \u4e09"
        arr[3] = "four: \u56db"
        arr[4] = 0
        buffer = arrayToBytes(arr)

        expected_length = 55
        expected = bytearray(expected_length)
        expected[0:4] = b"\x08\x00\x00\x00"
        expected[4:16] = b"one: \xe4\xb8\x80\x08\x00\x00\x00"
        expected[16:28] = b"two: \xe4\xba\x8c\n\x00\x00\x00"
        expected[28:42] = b"three: \xe4\xb8\x89\t\x00\x00\x00"
        expected[42:55] = b"four: \xe5\x9b\x9b\x00\x00\x00\x00"

        self.assertEqual(len(buffer), expected_length)

        self.assertEqual(buffer, expected)
        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (5,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # VLEN of bytes
        dt = special_dtype(vlen=bytes)
        arr = np.zeros((5,), dtype=dt)
        arr[0] = b"Parting"
        arr[1] = b"is such"
        arr[2] = b"sweet"
        arr[3] = b"sorrow"
        arr[4] = 0

        buffer = arrayToBytes(arr)

        expected = bytearray(45)
        expected[0:11] = b"\x07\x00\x00\x00Parting"
        expected[11:22] = b"\x07\x00\x00\x00is such"
        expected[22:31] = b"\x05\x00\x00\x00sweet"
        expected[31:41] = b"\x06\x00\x00\x00sorrow"
        expected[41:45] = b"\x00\x00\x00\x00"

        self.assertEqual(len(buffer), len(expected))
        self.assertEqual(buffer, expected)  # same serialization as with str

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (5,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        #
        # Compound str vlen
        #
        dt_vstr = special_dtype(vlen=str)
        dt = np.dtype([("x", "i4"), ("tag", dt_vstr), ("code", "S4")])
        arr = np.zeros((4,), dtype=dt)
        arr[0] = (42, "Hello", "X1")
        arr[3] = (84, "Bye", "XYZ")
        count = getByteArraySize(arr)
        buffer = arrayToBytes(arr)

        self.assertEqual(len(buffer), 56)
        self.assertEqual(buffer.find(b"Hello"), 8)
        self.assertEqual(buffer.find(b"Bye"), 49)
        self.assertEqual(buffer.find(b"X1"), 13)
        self.assertEqual(buffer.find(b"XYZ"), 52)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        #
        # Compound int vlen
        #
        dt_vint = special_dtype(vlen=np.dtype("<i4"))
        dt = np.dtype([("x", "int32"), ("tag", dt_vint)])
        arr = np.zeros((4,), dtype=dt)
        arr[0] = (42, np.array((), dtype="int32"))
        arr[3] = (84, np.array((1, 2, 3), dtype="int32"))
        count = getByteArraySize(arr)
        self.assertEqual(count, 44)
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 44)
        buffer_expected = {0: 42, 24: 84, 28: 12, 32: 1, 36: 2, 40: 3}
        for i in range(44):
            if i in buffer_expected:
                self.assertEqual(buffer[i], buffer_expected[i])
            else:
                self.assertEqual(buffer[i], 0)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        #
        # VLEN utf string with array type
        #
        dt_str = special_dtype(vlen=str)
        dt_arr_str = np.dtype((dt_str, (2,)))
        dt = np.dtype([("x", "i4"), ("tag", dt_arr_str)])
        arr = np.zeros((4,), dtype=dt)
        dt_str = special_dtype(vlen=str)
        arr[0] = (42, np.asarray(["hi", "bye"], dtype=dt_str))
        arr[3] = (84, np.asarray(["hi-hi", "bye-bye"], dtype=dt_str))
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 81)

        self.assertEqual(buffer.find(b"hi"), 8)
        self.assertEqual(buffer.find(b"bye"), 14)
        self.assertEqual(buffer.find(b"hi-hi"), 49)
        self.assertEqual(buffer.find(b"bye-bye"), 58)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,))

        self.assertEqual(arr.dtype, arr_copy.dtype)
        self.assertEqual(arr.shape, arr_copy.shape)
        self.assertTrue(ndarray_compare(arr, arr_copy))
        #
        # VLEN ascii with array type
        #
        dt_str = special_dtype(vlen=bytes)
        dt_arr_str = np.dtype((dt_str, (2,)))
        dt = np.dtype([("x", "i4"), ("tag", dt_arr_str)])
        arr = np.zeros((4,), dtype=dt)

        arr[0] = (42, np.asarray([b"hi", b"bye"], dtype=dt_str))
        arr[3] = (84, np.asarray([b"hi-hi", b"bye-bye"], dtype=dt_str))
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 81)

        self.assertEqual(buffer.find(b"hi"), 8)
        self.assertEqual(buffer.find(b"bye"), 14)
        self.assertEqual(buffer.find(b"hi-hi"), 49)
        self.assertEqual(buffer.find(b"bye-bye"), 58)
        # convert back to array

        arr_copy = bytesToArray(buffer, dt, (4,))
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # test Compound with VLEN
        count = 4
        fixed_str8_type = {
            "charSet": "H5T_CSET_ASCII",
            "class": "H5T_STRING",
            "length": 8,
            "strPad": "H5T_STR_NULLPAD",
        }
        fields = [
            {
                "type": {"class": "H5T_INTEGER", "base": "H5T_STD_U64BE"},
                "name": "VALUE1",
            },
            {
                "type": fixed_str8_type,
                "name": "VALUE2"
            },
            {
                "type": {
                    "class": "H5T_ARRAY",
                    "dims": [2],
                    "base": {
                        "class": "H5T_STRING",
                        "charSet": "H5T_CSET_ASCII",
                        "strPad": "H5T_STR_NULLTERM",
                        "length": "H5T_VARIABLE",
                    },
                },
                "name": "VALUE3",
            },
        ]

        datatype = {"class": "H5T_COMPOUND", "fields": fields}

        dt = createDataType(datatype)
        self.assertTrue(isVlen(dt))

        # create numpy vlen array
        arr = np.zeros((count,), dtype=dt)
        for i in range(count):
            e = arr[i]
            e["VALUE1"] = i + 1
            s = ""
            for j in range(i + 5):
                offset = (i + j) % 26
                s += chr(ord("A") + offset)
            e["VALUE2"] = s
            e["VALUE3"] = [b"Hi! " * (i + 1), b"Bye!" * (i + 1)]

        # converts to bytes
        data = arrayToBytes(arr)
        self.assertEqual(len(data), 192)  # will vary based on count

        # convert back to array
        arr_copy = bytesToArray(data, dt, (4,))

        self.assertEqual(arr.dtype, arr_copy.dtype)
        self.assertEqual(arr.shape, arr_copy.shape)
        for i in range(4):
            e = arr[i]
            e_copy = arr_copy[i]
            self.assertTrue(np.array_equal(e, e_copy))

    def testArrToBytesBase64(self):
        # Simple array
        dt = np.dtype("<i4")
        arr = np.asarray((1, 2, 3, 4), dtype=dt)
        buffer = arrayToBytes(arr, encoding="base64")
        # should be a bit longer than the byte representation...
        expected_num_bytes = np.prod(arr.shape) * dt.itemsize
        self.assertTrue(len(buffer) > expected_num_bytes)

        # convert buffer back to arr
        arr_copy = bytesToArray(buffer, dt, (4,), encoding="base64")
        self.assertTrue(np.array_equal(arr, arr_copy))

        # fixed length string
        dt = np.dtype("S8")
        arr = np.asarray(("abcdefgh", "ABCDEFGH", "12345678"), dtype=dt)
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (3,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # Compound non-vlen
        dt = np.dtype([("x", "f8"), ("y", "i4")])
        arr = np.zeros((4,), dtype=dt)
        arr[0] = (3.12, 42)
        arr[3] = (1.28, 69)
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # VLEN of int32's
        dt = special_dtype(vlen=np.dtype("<i4"))
        arr = np.zeros((4,), dtype=dt)
        arr[0] = np.int32([1, ])
        arr[1] = np.int32([1, 2])
        arr[2] = 0  # test un-initialized value
        arr[3] = np.int32([1, 2, 3])
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

        # VLEN of strings
        dt = special_dtype(vlen=str)
        arr = np.zeros((5,), dtype=dt)
        arr[0] = "one: \u4e00"
        arr[1] = "two: \u4e8c"
        arr[2] = "three: \u4e09"
        arr[3] = "four: \u56db"
        arr[4] = 0
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (5,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))
        # VLEN of bytes
        dt = special_dtype(vlen=bytes)
        arr = np.zeros((5,), dtype=dt)
        arr[0] = b"Parting"
        arr[1] = b"is such"
        arr[2] = b"sweet"
        arr[3] = b"sorrow"
        arr[4] = 0

        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (5,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

        #
        # Compound str vlen
        #
        dt_vstr = special_dtype(vlen=str)
        dt = np.dtype([("x", "i4"), ("tag", dt_vstr), ("code", "S4")])
        arr = np.zeros((4,), dtype=dt)
        arr[0] = (42, "Hello", "X1")
        arr[3] = (84, "Bye", "XYZ")
        count = getByteArraySize(arr)
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

        #
        # Compound int vlen
        #
        dt_vint = special_dtype(vlen=np.dtype("<i4"))
        dt = np.dtype([("x", "int32"), ("tag", dt_vint)])
        arr = np.zeros((4,), dtype=dt)
        arr[0] = (42, np.array((), dtype="int32"))
        arr[3] = (84, np.array((1, 2, 3), dtype="int32"))
        count = getByteArraySize(arr)
        self.assertEqual(count, 44)
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

        #
        # VLEN utf string with array type
        #
        dt_str = special_dtype(vlen=str)
        dt_arr_str = np.dtype((dt_str, (2,)))
        dt = np.dtype([("x", "i4"), ("tag", dt_arr_str)])
        arr = np.zeros((4,), dtype=dt)

        dt_str = special_dtype(vlen=str)
        arr[0] = (42, np.asarray(["hi", "bye"], dtype=dt_str))
        arr[3] = (84, np.asarray(["hi-hi", "bye-bye"], dtype=dt_str))
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,), encoding="base64")

        self.assertEqual(arr.dtype, arr_copy.dtype)
        self.assertEqual(arr.shape, arr_copy.shape)
        self.assertTrue(ndarray_compare(arr, arr_copy))
        #
        # VLEN ascii with array type
        #
        dt_str = special_dtype(vlen=bytes)
        dt_arr_str = np.dtype((dt_str, (2,)))
        dt = np.dtype([("x", "i4"), ("tag", dt_arr_str)])
        arr = np.zeros((4,), dtype=dt)

        dt_str = special_dtype(vlen=str)
        arr[0] = (42, np.asarray([b"hi", b"bye"], dtype=dt_str))
        arr[3] = (84, np.asarray([b"hi-hi", b"bye-bye"], dtype=dt_str))
        buffer = arrayToBytes(arr, encoding="base64")

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,), encoding="base64")
        self.assertTrue(ndarray_compare(arr, arr_copy))

    def testArrayCompareInt(self):
        # Simple array
        dt = np.dtype("<i4")
        arr1 = np.zeros((1024, 1024), dtype=dt)
        arr2 = np.zeros((1024, 1024), dtype=dt)
        for _ in range(100):
            self.assertTrue(ndarray_compare(arr1, arr2))
        arr1[123, 456] = 42
        self.assertFalse(ndarray_compare(arr1, arr2))

    def testArrayCompareVlenInt(self):
        # Vlen array
        dt_vint = special_dtype(vlen=np.dtype("<i4"))
        dt = np.dtype([("x", "int32"), ("tag", dt_vint)])
        arr1 = np.zeros((1024, 1024), dtype=dt)
        arr2 = np.zeros((1024, 1024), dtype=dt)
        e1 = (42, np.array((), dtype="int32"))
        e2 = (84, np.array((1, 2, 3), dtype="int32"))
        arr1[123, 456] = e1
        arr2[123, 456] = e1
        arr1[888, 999] = e2
        arr2[888, 999] = e2

        # performance is marginal for this case
        for _ in range(1):
            self.assertTrue(ndarray_compare(arr1, arr2))
        arr2[123, 456] = e2
        self.assertFalse(ndarray_compare(arr1, arr2))

    def testJsonToBytes(self):
        #
        # VLEN int
        #

        def array_equal(a, b):
            """ compare two values element by element."""
            if type(a) in (list, tuple, np.void, np.ndarray):
                if len(a) != len(b):
                    return False
                nelements = len(a)
                for i in range(nelements):
                    if not array_equal(a[i], b[i]):
                        return False
            else:
                # treat a string and bytes as equal if the utf-8 encoding
                # of the string is equal to the byte encoding
                if isinstance(a, str):
                    a = a.encode("utf8")
                if isinstance(b, str):
                    b = b.encode("utf8")
                # treat 0 and b"" as equivalent (uninitialized vlen)
                if not a and not b:
                    return True
                if a != b:
                    return False

            return True

        dt = special_dtype(vlen=np.dtype("int32"))
        shape = [4,]
        data = [
            [1,],
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
        ]
        arr = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(check_dtype(vlen=arr.dtype), np.dtype("int32"))
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 56)

        expected = bytearray(48)
        expected[0:8] = b"\x04\x00\x00\x00\x01\x00\x00\x00"
        expected[8:16] = b"\x08\x00\x00\x00\x01\x00\x00\x00"
        expected[16:24] = b"\x02\x00\x00\x00\x0c\x00\x00\x00"
        expected[24:32] = b"\x01\x00\x00\x00\x02\x00\x00\x00"
        expected[32:40] = b"\x03\x00\x00\x00\x10\x00\x00\x00"
        expected[40:48] = b"\x01\x00\x00\x00\x02\x00\x00\x00"
        expected[48:56] = b"\x03\x00\x00\x00\x04\x00\x00\x00"
        self.assertEqual(buffer, expected)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, shape)
        # np.array_equal doesn't work for object arrays
        self.assertEqual(arr.dtype, arr_copy.dtype)
        self.assertEqual(arr.shape, arr_copy.shape)
        for i in range(4):
            e = arr[i]
            e_copy = arr_copy[i]
            self.assertTrue(np.array_equal(e, e_copy))
        #
        # Compound vlen
        #
        dt_str = special_dtype(vlen=str)
        dt = np.dtype([("x", "i4"), ("tag", dt_str)])
        shape = [4, ]
        data = [[42, "Hello"], [0, 0], [0, 0], [84, "Bye"]]
        arr = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(tuple(arr[0]), (42, 'Hello'))
        self.assertEqual(tuple(arr[3]), (84, 'Bye'))
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 40)

        expected = bytearray(40)
        expected[0:10] = b'*\x00\x00\x00\x05\x00\x00\x00He'
        expected[10:20] = b'llo\x00\x00\x00\x00\x00\x00\x00'
        expected[20:30] = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00T'
        expected[30:40] = b'\x00\x00\x00\x03\x00\x00\x00Bye'
        self.assertEqual(buffer, expected)

        # convert back to array
        arr_copy = bytesToArray(buffer, dt, (4,))
        # np.array_equal doesn't work for object arrays
        self.assertEqual(arr.dtype, arr_copy.dtype)
        self.assertEqual(arr.shape, arr_copy.shape)
        self.assertTrue(array_equal(arr, arr_copy))

        #
        # VLEN utf with array type
        #
        dt_str = special_dtype(vlen=str)
        dt_arr_str = np.dtype((dt_str, (2,)))
        dt = np.dtype([("x", "i4"), ("tag", dt_arr_str)])
        shape = [4,]
        data = [
            [42, ["hi", "bye"]],
            [0, [0, 0]],
            [0, [0, 0]],
            [84, ["hi-hi", "bye-bye"]],
        ]
        arr = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(arr, np.ndarray))
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 81)
        self.assertEqual(buffer.find(b"hi"), 8)
        self.assertEqual(buffer.find(b"bye"), 14)
        self.assertEqual(buffer.find(b"hi-hi"), 49)
        self.assertEqual(buffer.find(b"bye-bye"), 58)
        arr_copy = bytesToArray(buffer, dt, shape)

        self.assertEqual(arr.dtype, arr_copy.dtype)
        self.assertEqual(arr.shape, arr_copy.shape)
        self.assertTrue(array_equal(e, e_copy))

        #
        # VLEN ascii with array type
        #
        dt_str = special_dtype(vlen=bytes)
        dt_arr_str = np.dtype((dt_str, (2,)))
        dt = np.dtype([("x", "i4"), ("tag", dt_arr_str)])
        shape = [4,]
        data = [
            [42, [b"hi", b"bye"]],
            [0, [0, 0]],
            [0, [0, 0]],
            [84, [b"hi-hi", b"bye-bye"]],
        ]
        arr = jsonToArray(shape, dt, data)
        self.assertTrue(isinstance(arr, np.ndarray))
        buffer = arrayToBytes(arr)
        self.assertEqual(len(buffer), 81)
        self.assertEqual(buffer.find(b"hi"), 8)
        self.assertEqual(buffer.find(b"bye"), 14)
        self.assertEqual(buffer.find(b"hi-hi"), 49)
        self.assertEqual(buffer.find(b"bye-bye"), 58)
        arr_copy = bytesToArray(buffer, dt, shape)

        self.assertEqual(arr.dtype, arr_copy.dtype)
        self.assertEqual(arr.shape, arr_copy.shape)
        self.assertTrue(array_equal(e, e_copy))

    def testIndexIterator(self):
        i = 0
        for index in IndexIterator((10,)):
            self.assertEqual(index, (i,))
            i += 1
        self.assertEqual(i, 10)
        i = 0
        for index in IndexIterator((10,), sel=slice(0, 10, 2)):
            self.assertEqual(index, (i,))

            i += 2
        self.assertEqual(i, 10)
        i = 2
        for index in IndexIterator((10, ), sel=slice(2, 8)):
            self.assertEqual(index, (i,))
            i += 1
        self.assertEqual(i, 8)
        cnt = 0
        for index in IndexIterator((4, 5)):
            cnt += 1
        self.assertEqual(cnt, 20)
        cnt = 0
        for index in IndexIterator((8, 10), sel=(slice(0, 8, 2), slice(0, 10, 2))):
            cnt += 1
        self.assertEqual(cnt, 20)

    def testGetNumpyValue(self):
        # test int conversion
        dt = np.dtype("<i4")
        val = getNumpyValue(42, dt=dt)
        self.assertTrue(isinstance(val, np.int32))
        self.assertEqual(42, val)

        # test fixed length string conversion
        dt = np.dtype("S5")
        val = getNumpyValue("hello", dt=dt)
        self.assertTrue(isinstance(val, np.bytes_))
        self.assertEqual(val, b"hello")

        # test variable length string conversion
        dt = special_dtype(vlen=bytes)
        val = getNumpyValue("hello", dt=dt)
        self.assertTrue(isinstance(val, str))
        self.assertEqual(val, "hello")

        # test compound type
        dt = np.dtype([('int', "<i4"), ('str', "S4")])
        val = getNumpyValue((42, "hdf5"), dt=dt)
        self.assertTrue(isinstance(val, np.void))
        self.assertEqual(val[0], 42)
        self.assertEqual(val[1], b'hdf5')

        # test array of ints
        dt = np.dtype("<i4")
        arr = np.array([0, 1], dtype=dt)
        dt = np.dtype(("<i4", (len(arr),)))
        val = getNumpyValue(arr, dt=dt)

        self.assertTrue(np.array_equal(val, arr))
        self.assertTrue(isinstance(val[0], np.int32))

        # test array of floats
        dt = np.dtype("f4")
        arr = np.array([0.001, 1.001], dtype=dt)
        val = getNumpyValue(arr, dt=np.dtype(("f4", (len(arr),))))

        self.assertTrue(np.array_equal(val, arr))
        self.assertTrue(isinstance(val[0], np.float32))

        # test array of fixed-length strings
        dt = np.dtype("S5")
        arr = np.array([b'hello', b'world'], dtype=dt)
        val = getNumpyValue(arr, dt=np.dtype(("S5", (len(arr),))))

        self.assertTrue(np.array_equal(val, arr))
        self.assertTrue(isinstance(val[0], np.bytes_))

        # test nan string
        dt = np.dtype("f4")
        val = getNumpyValue("nan", dt=dt)
        self.assertTrue(isinstance(val, np.float32))
        self.assertTrue(val != val)

    def testGetNumpyValueBase64Encoded(self):
        # Set up value, numpy dtype, and expected type after decoding
        value_info = []
        value_info.append([42, np.dtype("<i4"), np.int32])  # int
        value_info.append([1.001, np.dtype("f4"), np.float32])  # float
        value_info.append([b"hello", np.dtype("S5"), np.bytes_])  # fixed-length string
        value_info.append([(42, b'hdf5'),
                           np.dtype([('int', "<i4"), ('str', "S4")]), np.void])  # compound type
        np_values = []

        for vi in value_info:
            np_values.append(np.array(vi[0], dtype=vi[1]))

        for i in range(len(np_values)):
            numpy_dtype_out = value_info[i][2]

            # Turn numpy array to bytes object which can be encoded
            encoded_val = np_values[i].tobytes()
            # Encode numpy bytes object
            encoded_val = base64.b64encode(encoded_val)
            # Decode from bytes object to regular string containing a base64 encoded numpy array
            # This prevents the utf-8 encoding inside getNumpyValue from prepending b'
            encoded_val = encoded_val.decode()
            decoded_val = getNumpyValue(encoded_val, dt=np_values[i].dtype, encoding="base64")
            self.assertTrue(isinstance(decoded_val, numpy_dtype_out))
            self.assertEqual(decoded_val, np_values[i])

        # test array types

        # Set up value, numpy dtype, and expected type after decoding
        value_info = []
        value_info.append([np.array([0, 1], dtype=np.dtype("<i4")),
                           np.dtype(("<i4", (2,))), np.int32])  # int array
        value_info.append([np.array([0.001, 1.001], dtype=np.dtype("f4")),
                           np.dtype(("f4", (2,))), np.float32])  # float array
        value_info.append([np.array([b'hello', b'world'], dtype=np.dtype("S5")),
                           np.dtype(("S5", (2,))), np.bytes_])  # fixed length string array

        for i in range(len(value_info)):
            this_array = value_info[i][0]
            array_dtype = value_info[i][1]
            array_dtype_out = value_info[i][2]

            # Turn numpy array to bytes object which can be encoded
            encoded_val = this_array.tobytes()
            # Encode numpy bytes object
            encoded_val = base64.b64encode(encoded_val)
            # Decode from bytes object to regular string containing a base64 encoded numpy array
            # This prevents the utf-8 encoding inside getNumpyValue from prepending b'
            encoded_val = encoded_val.decode()
            decoded_val = getNumpyValue(encoded_val, dt=array_dtype, encoding="base64")

            self.assertTrue(np.array_equal(decoded_val, this_array))
            self.assertTrue(isinstance(decoded_val[0], array_dtype_out))

        # test invalid base64 length
        try:
            dt = np.dtype("<i8")
            getNumpyValue("KgAAAA==", dt=dt, encoding="base64")
            self.assertTrue(False)
        except ValueError:
            pass  # expected

    def testJsonToArrayOnNoneArray(self):
        data_dtype = np.dtype("i4")
        data_shape = [3, ]
        data_json = [None, None, None]
        arr = None
        try:
            arr = jsonToArray(data_shape, data_dtype, data_json)
        except Exception as e:
            print(f"Exception while testing jsonToArray on array with None elements: {e}")
        self.assertEqual(arr.shape, (3, ))
        self.assertTrue(arr.dtype == data_dtype)

    def testGetBroadcastShape(self):
        bcshape = getBroadcastShape([1, ], 1)
        self.assertEqual(bcshape, None)
        bcshape = getBroadcastShape([2, 3], 6)
        self.assertEqual(bcshape, None)
        bcshape = getBroadcastShape([2, 3], 5)
        self.assertEqual(bcshape, None)

        bcshape = getBroadcastShape([4, 5], 1)
        self.assertEqual(bcshape, [1, ])
        bcshape = getBroadcastShape([4, 5], 5)
        self.assertEqual(bcshape, [5, ])

        bcshape = getBroadcastShape([2, 3, 5], 1)
        self.assertEqual(bcshape, [1, ])
        bcshape = getBroadcastShape([2, 3, 5], 5)
        self.assertEqual(bcshape, [5, ])
        bcshape = getBroadcastShape([2, 3, 5], 15)
        self.assertEqual(bcshape, [3, 5])

    def testJsonToArrayOnNoneCompoundArray(self):
        # compound type
        dt = np.dtype([("a", "i4"), ("b", "S5")])
        shape = [1,]
        data = None

        arr = jsonToArray(shape, dt, data)

        self.assertEqual(arr.shape, (1,))
        self.assertEqual(arr.dtype, dt)


if __name__ == "__main__":
    # setup test files

    unittest.main()
