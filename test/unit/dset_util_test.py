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
import logging

from h5json.filters import getFilterItem
from h5json.dset_util import guessChunk, shrinkChunk, getChunkSize, expandChunk, generateLayout
from h5json.dset_util import getDatasetLayoutClass, getContiguousLayout, getChunkDims
from h5json.dset_util import validateLayout, validateDatasetCreationProps, getDatasetLayout
from h5json.dset_util import getFillValue, generate_dcpl
from h5json.objid import createObjId


class DsetUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DsetUtilTest, self).__init__(*args, **kwargs)
        # main
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testGetLayout(self):
        contiguous_layout = {'class': 'H5D_CONTIGUOUS'}
        fixed_1d_shape_json = {'class': 'H5S_SIMPLE', 'dims': [10]}
        resizable_shape_json = {'class': 'H5S_SIMPLE', 'dims': [10], 'maxdims': [20]}
        base_type = 'H5T_IEEE_F32LE'
        type_json = {'class': 'H5T_FLOAT', 'base': base_type}
        chunked_layout = {'class': 'H5D_CHUNKED', 'dims': [2, ]}
        cpl = {'fillValue': 3.12, 'layout': contiguous_layout}

        dset_json = {'id': 'd-f4a9f95e-c8962a53-f6c8-f18440-78d051',
                     'root': 'g-f4a9f95e-c8962a53-7c21-71d640-1ea2db',
                     'created': 1760613930.3584619,
                     'type': type_json,
                     'shape': resizable_shape_json,
                     'lastModified': 1760613930.3584619,
                     'creationProperties': cpl}

        layout = getDatasetLayout(dset_json)
        self.assertTrue("class" in layout)
        layout_class = getDatasetLayoutClass(dset_json)
        self.assertEqual(layout_class, "H5D_CONTIGUOUS")

        # contigous layout with resizable shape should raise exception
        try:
            validateLayout(dset_json["shape"], type_json, layout)
            self.assertTrue(False)  # should not reach here
        except ValueError:
            pass  # should raise exception

        dset_json["shape"] = fixed_1d_shape_json
        layout = getDatasetLayout(dset_json)
        self.assertTrue("class" in layout)
        layout_class = getDatasetLayoutClass(dset_json)
        self.assertEqual(layout_class, "H5D_CONTIGUOUS")

        dset_json["shape"] = resizable_shape_json
        cpl["layout"] = chunked_layout
        layout = getDatasetLayout(dset_json)
        self.assertTrue("class" in layout)
        layout_class = getDatasetLayoutClass(dset_json)
        self.assertEqual(layout_class, "H5D_CHUNKED")

        try:
            validateLayout(dset_json["shape"], type_json, layout)
        except ValueError:
            self.assertTrue(False)  # shouldn't raise exception

        chunk_dims = getChunkDims(dset_json)
        self.assertEqual(chunk_dims, (2, ))

        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
        except ValueError:
            self.assertTrue(False)  # shouldn't raise exception

    def testFilterValidation(self):

        shape_json = {'class': 'H5S_SIMPLE', 'dims': [500]}
        base_type = 'H5T_IEEE_F32LE'
        type_json = {'class': 'H5T_FLOAT', 'base': base_type}
        contiguous_layout = {'class': 'H5D_CONTIGUOUS'}
        chunked_layout = {'class': 'H5D_CHUNKED', 'dims': [100, ]}
        deflate_filter = {'class': 'H5Z_FILTER_DEFLATE', 'id': 1, 'name': 'deflate'}
        filters = [deflate_filter, ]
        cpl = {'fillValue': 3.12, 'layout': contiguous_layout, "filters": filters}

        dset_json = {'id': 'd-f4a9f95e-c8962a53-f6c8-f18440-78d051',
                     'root': 'g-f4a9f95e-c8962a53-7c21-71d640-1ea2db',
                     'created': 1760613930.3584619,
                     'type': type_json,
                     'shape': shape_json,
                     'lastModified': 1760613930.3584619,
                     'creationProperties': cpl}

        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
            self.assertTrue(False)  # should not reach here
        except ValueError:
            pass  # filters are invalid with contiguous layout

        cpl["layout"] = chunked_layout
        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
        except ValueError:
            self.assertTrue(False)  # shouldn't raise exception

        # add an invlaid level option for deflate
        deflate_filter["level"] = 20
        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
            self.assertTrue(False)  # should not reach here
        except ValueError:
            pass  # invalid deflate level

        deflate_filter["level"] = 5
        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
        except ValueError:
            self.assertTrue(False)  # shouldn't raise exception

        # try with just a filter name
        gzip_filter = getFilterItem("gzip")
        cpl["filters"] = [gzip_filter, ]
        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
        except ValueError:
            self.assertTrue(False)  # shouldn't raise exception

        # try with an invalid filter name
        cpl["filters"] = ["invalid_filter_name", ]
        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
            self.assertTrue(False)  # should not reach here
        except ValueError:
            pass  # invalid filter name

        deflate_filter = {'class': 'H5Z_FILTER_DEFLATE', 'id': 1, 'level': 9, 'name': 'deflate'}
        fletcher_filter = {'class': 'H5Z_FILTER_FLETCHER32', 'id': 3, 'name': 'fletcher32'}
        filters = [fletcher_filter, deflate_filter]
        cpl["filters"] = filters
        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
        except ValueError:
            self.assertTrue(False)  # shouldn't raise exception

        sc_filter = {'class': 'H5Z_FILTER_SCALEOFFSET', 'id': 6, 'name': 'scaleoffset'}
        sc_filter['scaleOffset'] = 12
        sc_filter['scaleType'] = 'H5Z_SO_INT'
        filters = [sc_filter, ]
        cpl["filters"] = filters
        try:
            validateDatasetCreationProps(cpl, type_json, dset_json["shape"])
        except ValueError:
            self.assertTrue(False)  # shouldn't raise exception

    def testGuessChunk(self):

        typesize = "H5T_VARIABLE"
        logging.debug("hello")

        shape = {"class": "H5S_NULL"}
        layout = guessChunk(shape, typesize)
        self.assertTrue(layout is None)

        shape = {"class": "H5S_SCALAR"}
        layout = guessChunk(shape, typesize)
        self.assertEqual(layout, (1,))

        shape = {"class": "H5S_SIMPLE", "dims": [100, 100]}
        layout = guessChunk(shape, typesize)
        self.assertTrue(len(layout), 2)
        for i in range(2):
            self.assertTrue(layout[i] >= 1)
            self.assertTrue(layout[i] <= 100)

        typesize = 8
        layout = guessChunk(shape, typesize)
        self.assertTrue(len(layout), 2)
        for i in range(2):
            self.assertTrue(layout[i] >= 1)
            self.assertTrue(layout[i] <= 100)

        shape = {"class": "H5S_SIMPLE", "dims": [5]}
        layout = guessChunk(shape, typesize)
        self.assertEqual(layout, (5,))

        shape = {"class": "H5S_SIMPLE", "dims": [100, 100, 100]}
        chunk_max = 400
        layout = guessChunk(shape, typesize, chunk_max=chunk_max)
        self.assertTrue(len(layout), 3)
        for i in range(3):
            self.assertTrue(layout[i] >= 1)
            self.assertTrue(layout[i] < 100)
        chunk_size = getChunkSize(layout, typesize)
        self.assertTrue(chunk_size <= chunk_max)

        shape = {"class": "H5S_SIMPLE", "dims": [100, 0], "maxdims": [100, "H5S_UNLIMITED"]}
        layout = guessChunk(shape, typesize)
        self.assertTrue(len(layout), 2)
        for i in range(2):
            self.assertTrue(layout[i] >= 1)
            self.assertTrue(layout[i] <= 1024)

        dims = [50000, 80000]
        shape = {'class': 'H5S_SIMPLE', 'dims': dims}
        chunk_min = 1048576
        chunk_max = 4194304
        layout = guessChunk(shape, typesize, chunk_min=chunk_min, chunk_max=chunk_max)
        self.assertTrue(len(layout), 2)
        self.assertTrue(layout[0] < dims[0])
        self.assertTrue(layout[1] < dims[1])
        chunk_size = layout[0] * layout[1] * typesize
        self.assertTrue(chunk_size >= chunk_min)
        self.assertTrue(chunk_size <= chunk_max)

        shape = {"class": "H5S_SCALAR"}
        layout = guessChunk(shape, typesize)
        self.assertEqual(layout, (1,))

        shape = {"class": "H5S_NULL"}
        layout = guessChunk(shape, typesize)
        self.assertEqual(layout, None)

    def testShrinkChunk(self):
        CHUNK_MIN = 500
        CHUNK_MAX = 5000
        typesize = 1
        layout = (1, 2, 3)
        shrunk = shrinkChunk(layout, typesize, chunk_max=CHUNK_MAX)
        self.assertEqual(shrunk, layout)

        layout = (100, 200, 300)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes > CHUNK_MAX)
        shrunk = shrinkChunk(layout, typesize, chunk_max=CHUNK_MAX)
        rank = len(layout)
        for i in range(rank):
            self.assertTrue(shrunk[i] >= 1)
            self.assertTrue(shrunk[i] <= 1000 * (i + 1))
        num_bytes = getChunkSize(shrunk, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

        layout = (300, 200, 100)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes > CHUNK_MAX)
        shrunk = shrinkChunk(layout, typesize, chunk_max=CHUNK_MAX)
        rank = len(layout)
        for i in range(rank):
            self.assertTrue(shrunk[i] >= 1)
            self.assertTrue(shrunk[i] <= 1000 * (3 - i))
        num_bytes = getChunkSize(shrunk, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

        CHUNK_MIN = 1 * 1024 * 1024
        CHUNK_MAX = 4 * 1024 * 1024
        typesize = 4
        layout = (117, 201, 189, 1)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes > CHUNK_MAX)
        shrunk = shrinkChunk(layout, typesize, chunk_max=CHUNK_MAX)
        self.assertEqual(shrunk, (59, 101, 95, 1))
        num_bytes = getChunkSize(shrunk, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

        shape = {
            "class": "H5S_SIMPLE",
            "dims": [50000, 80000],
        }
        layout = [782, 125]
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

    def testExpandChunk(self):
        CHUNK_MIN = 5000
        CHUNK_MAX = 50000

        typesize = 20
        shape = {"class": "H5S_SIMPLE", "dims": [12, ], "maxdims": [20, ]}
        layout = (20,)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        # chunk layout can't be larger than dataspace
        self.assertTrue(num_bytes < CHUNK_MIN)
        self.assertEqual(expanded, (20,))

        typesize = 1
        shape = {"class": "H5S_SIMPLE", "dims": [10, 10, 10]}
        layout = (10, 10, 10)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        # chunk layout can't be larger than dataspace
        self.assertTrue(num_bytes < CHUNK_MIN)
        self.assertEqual(expanded, (10, 10, 10))

        shape = {"class": "H5S_SIMPLE", "dims": [1000, 2000, 3000]}
        layout = (10, 10, 10)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

        shape = {
            "class": "H5S_SIMPLE",
            "dims": [1000, 10, 1000],
            "maxdims": [1000, 100, 1000],
        }
        layout = (10, 10, 10)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

        shape = {
            "class": "H5S_SIMPLE",
            "dims": [1000, 0, 1000],
            "maxdims": [1000, 100, 1000],
        }
        layout = (10, 10, 10)
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

        shape = {
            "class": "H5S_SIMPLE",
            "dims": [1000, 10, 1000],
            "maxdims": [1000, "H5S_UNLIMITED", 1000],
        }
        layout = (10, 10, 10)
        typesize = 4
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

        CHUNK_MIN = 1024 * 1024
        CHUNK_MAX = 4 * CHUNK_MIN
        shape = {
            "class": "H5S_SIMPLE",
            "dims": [50000, 80000],
        }
        layout = [100, 100]
        typesize = 4
        num_bytes = getChunkSize(layout, typesize)
        self.assertTrue(num_bytes < CHUNK_MIN)
        expanded = expandChunk(layout, typesize, shape, chunk_min=CHUNK_MIN)
        num_bytes = getChunkSize(expanded, typesize)
        self.assertTrue(num_bytes > CHUNK_MIN)
        self.assertTrue(num_bytes < CHUNK_MAX)

    def testGenerateLayout(self):
        chunk_min = 4000
        chunk_max = 8000
        shape = {
            "class": "H5S_SIMPLE",
            "dims": [40, 20],
        }
        base_type = 'H5T_IEEE_F32LE'
        type_json = {'class': 'H5T_FLOAT', 'base': base_type}

        kwargs = {"chunk_min": chunk_min, "chunk_max": chunk_max}
        layout = generateLayout(shape, type_json, **kwargs)
        self.assertTrue("class" in layout)
        self.assertEqual(layout["class"], "H5D_CONTIGUOUS")
        self.assertFalse("dims" in layout)

        layout = generateLayout(shape, type_json, chunks=True, **kwargs)
        self.assertTrue("class" in layout)
        self.assertEqual(layout["class"], "H5D_CHUNKED")
        self.assertTrue("dims" in layout)
        self.assertEqual(layout["dims"], [40, 20])

        layout = generateLayout(shape, type_json, chunks=(20, 10), **kwargs)
        self.assertTrue("class" in layout)
        self.assertEqual(layout["class"], "H5D_CHUNKED")
        self.assertTrue("dims" in layout)
        self.assertEqual(layout["dims"], [20, 10])

        try:
            # proposed chunk shape can't be larger than shape in
            # any dimension
            generateLayout(shape, type_json, chunks=(50, 10), **kwargs)
            self.assertTrue(False)  # shouldn't get here
        except ValueError:
            pass  # expected

        shape = {
            "class": "H5S_SIMPLE",
            "dims": [0, 20],
            "maxdims": [0, 20]
        }
        layout = generateLayout(shape, type_json, **kwargs)
        self.assertTrue("class" in layout)
        self.assertEqual(layout["class"], "H5D_CHUNKED")
        self.assertTrue("dims" in layout)
        dims = layout["dims"]
        self.assertEqual(len(dims), 2)
        self.assertTrue(dims[0] > 0)
        self.assertTrue(dims[1] > 0)

    def testGetContiguousLayout(self):
        typesize = 4
        chunk_min = 400
        chunk_max = 800

        kwargs = {"chunk_min": chunk_min, "chunk_max": chunk_max}

        def get_num_bytes(dims):
            num_bytes = typesize
            for n in dims:
                num_bytes *= n
            return num_bytes

        try:
            shape = {"class": "H5S_SIMPLE", "dims": [100, 100]}
            layout = getContiguousLayout(shape, "H5T_VARIABLE", **kwargs)
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        shape = {"class": "H5S_NULL"}
        layout = getContiguousLayout(shape, typesize, **kwargs)
        self.assertTrue(layout is None)

        shape = {"class": "H5S_SCALAR"}
        layout = getContiguousLayout(shape, typesize, **kwargs)
        self.assertEqual(layout, (1,))

        for extent in (1, 100, 10000):
            dims = [
                extent,
            ]
            shape = {"class": "H5S_SIMPLE", "dims": dims}
            layout = getContiguousLayout(shape, typesize, **kwargs)
            self.assertTrue(len(layout), 1)
            chunk_bytes = get_num_bytes(layout)
            space_bytes = get_num_bytes(dims)
            if space_bytes > chunk_min:
                self.assertTrue(chunk_bytes >= chunk_min)

            self.assertTrue(chunk_bytes <= chunk_max)

        for extent in (1, 9, 90):
            dims = [extent, extent]
            shape = {"class": "H5S_SIMPLE", "dims": dims}
            layout = getContiguousLayout(shape, typesize, **kwargs)
            self.assertTrue(len(layout), 2)
            for i in range(2):
                self.assertTrue(layout[i] >= 1)
                self.assertTrue(layout[i] <= extent)
            self.assertEqual(layout[1], extent)
            chunk_bytes = get_num_bytes(layout)
            space_bytes = get_num_bytes(dims)

            if space_bytes > chunk_min:
                self.assertTrue(chunk_bytes >= chunk_min)
            self.assertTrue(chunk_bytes <= chunk_max)

        for extent in (1, 10, 100):
            dims = [extent, extent, 50]
            shape = {"class": "H5S_SIMPLE", "dims": dims}
            layout = getContiguousLayout(shape, typesize, **kwargs)
            self.assertTrue(len(layout), 3)
            for i in range(3):
                self.assertTrue(layout[i] >= 1)
                self.assertTrue(layout[i] <= dims[i])

            chunk_bytes = get_num_bytes(layout)
            space_bytes = get_num_bytes(dims)

            if space_bytes > chunk_min:
                self.assertTrue(chunk_bytes >= chunk_min)
            self.assertTrue(chunk_bytes <= chunk_max)

        for extent in (1, 100, 1000):
            dims = [extent, 4]
            shape = {"class": "H5S_SIMPLE", "dims": dims}
            layout = getContiguousLayout(shape, typesize, **kwargs)
            self.assertTrue(len(layout), 2)
            for i in range(2):
                self.assertTrue(layout[i] >= 1)
                self.assertTrue(layout[i] <= dims[i])

            chunk_bytes = get_num_bytes(layout)
            space_bytes = get_num_bytes(dims)

            if space_bytes > chunk_min:
                self.assertTrue(chunk_bytes >= chunk_min)
            self.assertTrue(chunk_bytes <= chunk_max)

    def testGetFillValue(self):
        obj_json = {"creationProperties": {"fillValue": 42}}
        self.assertEqual(getFillValue(obj_json), 42)

        # also accept being passed the cpl dict directly
        cpl = {"fillValue": 42}
        self.assertEqual(getFillValue(cpl), 42)

        # no fill value set
        self.assertEqual(getFillValue({"creationProperties": {}}), None)
        self.assertEqual(getFillValue({}), None)

    def testValidateLayoutChunkDims(self):
        type_json = {"class": "H5T_INTEGER", "base": "H5T_STD_I32LE"}
        shape = {"class": "H5S_SIMPLE", "dims": [10, 20]}

        # layout rank must match shape rank
        try:
            validateLayout(shape, type_json, {"class": "H5D_CHUNKED", "dims": [5]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # chunk dims must be integers
        try:
            validateLayout(shape, type_json, {"class": "H5D_CHUNKED", "dims": [5, "x"]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # chunk extent must be positive
        try:
            validateLayout(shape, type_json, {"class": "H5D_CHUNKED", "dims": [5, 0]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # without maxdims, chunk extent can't exceed the shape extent
        try:
            validateLayout(shape, type_json, {"class": "H5D_CHUNKED", "dims": [5, 30]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # with a fixed (non-extensible) maxdims, chunk extent can't exceed it
        shape_ext = {"class": "H5S_SIMPLE", "dims": [10, 20], "maxdims": [10, 25]}
        try:
            validateLayout(shape_ext, type_json, {"class": "H5D_CHUNKED", "dims": [5, 30]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # an unlimited maxdims dimension allows any positive chunk extent
        shape_unlim = {"class": "H5S_SIMPLE", "dims": [10, 20], "maxdims": [10, "H5S_UNLIMITED"]}
        validateLayout(shape_unlim, type_json, {"class": "H5D_CHUNKED", "dims": [5, 1000]})

        # a single int chunk dim gets promoted to a 1-element list
        shape_1d = {"class": "H5S_SIMPLE", "dims": [10]}
        validateLayout(shape_1d, type_json, {"class": "H5D_CHUNKED", "dims": 5})

        # missing "class" key
        try:
            validateLayout(shape, type_json, {"dims": [5, 10]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # unrecognized layout class
        try:
            validateLayout(shape, type_json, {"class": "H5D_BOGUS"})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

    def testValidateLayoutRefClasses(self):
        type_json = {"class": "H5T_INTEGER", "base": "H5T_STD_I32LE"}
        vlen_type = {
            "class": "H5T_STRING",
            "charSet": "H5T_CSET_ASCII",
            "length": "H5T_VARIABLE",
            "strPad": "H5T_STR_NULLTERM",
        }
        shape = {"class": "H5S_SIMPLE", "dims": [10, 20]}
        shape_ext = {"class": "H5S_SIMPLE", "dims": [10, 20], "maxdims": [10, 25]}

        # H5D_CONTIGUOUS_REF
        try:
            validateLayout(
                shape, vlen_type,
                {"class": "H5D_CONTIGUOUS_REF", "file_uri": "f", "offset": 0, "size": 100}
            )
            self.assertTrue(False)  # variable-length types not allowed
        except ValueError:
            pass  # expected

        for missing_key in ("file_uri", "offset", "size"):
            layout = {"class": "H5D_CONTIGUOUS_REF", "file_uri": "f", "offset": 0, "size": 100}
            del layout[missing_key]
            try:
                validateLayout(shape, type_json, layout)
                self.assertTrue(False)
            except ValueError:
                pass  # expected

        try:
            layout = {
                "class": "H5D_CONTIGUOUS_REF", "file_uri": "f", "offset": 0, "size": 100,
                "dims": [5, 10],
            }
            validateLayout(shape, type_json, layout)
            self.assertTrue(False)  # dims not allowed for this layout class
        except ValueError:
            pass  # expected

        try:
            layout = {"class": "H5D_CONTIGUOUS_REF", "file_uri": "f", "offset": 0, "size": 100}
            validateLayout(shape_ext, type_json, layout)
            self.assertTrue(False)  # maxdims not allowed for this layout class
        except ValueError:
            pass  # expected

        # valid H5D_CONTIGUOUS_REF
        layout = {"class": "H5D_CONTIGUOUS_REF", "file_uri": "f", "offset": 0, "size": 100}
        validateLayout(shape, type_json, layout)

        # H5D_CHUNKED_REF
        try:
            layout = {"class": "H5D_CHUNKED_REF", "file_uri": "f", "dims": [5, 10], "chunks": {}}
            validateLayout(shape, vlen_type, layout)
            self.assertTrue(False)  # variable-length types not allowed
        except ValueError:
            pass  # expected

        for missing_key in ("file_uri", "dims", "chunks"):
            layout = {"class": "H5D_CHUNKED_REF", "file_uri": "f", "dims": [5, 10], "chunks": {}}
            del layout[missing_key]
            try:
                validateLayout(shape, type_json, layout)
                self.assertTrue(False)
            except ValueError:
                pass  # expected

        # valid H5D_CHUNKED_REF
        layout = {"class": "H5D_CHUNKED_REF", "file_uri": "f", "dims": [5, 10], "chunks": {}}
        validateLayout(shape, type_json, layout)

        # H5D_CHUNKED_REF_INDIRECT
        root_id = createObjId("groups")
        chunk_table_id = createObjId("datasets", root_id=root_id)

        try:
            layout = {"class": "H5D_CHUNKED_REF_INDIRECT", "dims": [5, 10], "chunk_table": chunk_table_id}
            validateLayout(shape, vlen_type, layout)
            self.assertTrue(False)  # variable-length types not allowed
        except ValueError:
            pass  # expected

        for missing_key in ("dims", "chunk_table"):
            layout = {"class": "H5D_CHUNKED_REF_INDIRECT", "dims": [5, 10], "chunk_table": chunk_table_id}
            del layout[missing_key]
            try:
                validateLayout(shape, type_json, layout)
                self.assertTrue(False)
            except ValueError:
                pass  # expected

        try:
            layout = {"class": "H5D_CHUNKED_REF_INDIRECT", "dims": [5, 10], "chunk_table": "bogus-id"}
            validateLayout(shape, type_json, layout)
            self.assertTrue(False)  # invalid chunk table uuid
        except ValueError:
            pass  # expected

        # valid H5D_CHUNKED_REF_INDIRECT
        layout = {"class": "H5D_CHUNKED_REF_INDIRECT", "dims": [5, 10], "chunk_table": chunk_table_id}
        validateLayout(shape, type_json, layout)

    def testValidateLayoutStorageClasses(self):
        type_json = {"class": "H5T_INTEGER", "base": "H5T_STD_I32LE"}
        shape = {"class": "H5S_SIMPLE", "dims": [10, 20]}
        shape_ext = {"class": "H5S_SIMPLE", "dims": [10, 20], "maxdims": [10, 25]}

        # H5D_CHUNKED requires a "dims" key
        try:
            validateLayout(shape, type_json, {"class": "H5D_CHUNKED"})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # H5D_CHUNKED is only valid with an H5S_SIMPLE shape class
        try:
            scalar_shape_with_dims = {"class": "H5S_SCALAR", "dims": [1]}
            validateLayout(scalar_shape_with_dims, type_json, {"class": "H5D_CHUNKED", "dims": [1]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # H5D_CONTIGUOUS: "dims" not allowed in layout
        try:
            validateLayout(shape, type_json, {"class": "H5D_CONTIGUOUS", "dims": [5, 10]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # H5D_CONTIGUOUS: "maxdims" not allowed in shape
        try:
            validateLayout(shape_ext, type_json, {"class": "H5D_CONTIGUOUS"})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # valid H5D_CONTIGUOUS
        validateLayout(shape, type_json, {"class": "H5D_CONTIGUOUS"})

        # H5D_COMPACT: "dims" not allowed in layout
        try:
            validateLayout(shape, type_json, {"class": "H5D_COMPACT", "dims": [5, 10]})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # H5D_COMPACT: "maxdims" not allowed in shape
        try:
            validateLayout(shape_ext, type_json, {"class": "H5D_COMPACT"})
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # valid H5D_COMPACT
        validateLayout(shape, type_json, {"class": "H5D_COMPACT"})

    def testGenerateDcpl(self):
        type_json = {"class": "H5T_INTEGER", "base": "H5T_STD_I32LE"}

        # H5S_NULL / H5S_SCALAR (non-simple) shapes: creation property
        # list is trivially empty
        null_shape = {"class": "H5S_NULL"}
        self.assertEqual(generate_dcpl(null_shape, type_json), {})

        scalar_shape = {"class": "H5S_SCALAR"}
        self.assertEqual(generate_dcpl(scalar_shape, type_json), {})

        # chunks/filters aren't supported for non-simple shapes
        try:
            generate_dcpl(null_shape, type_json, chunks=(2,))
            self.assertTrue(False)
        except TypeError:
            pass  # expected

        deflate_filter = {"class": "H5Z_FILTER_DEFLATE", "id": 1, "name": "deflate"}
        try:
            generate_dcpl(scalar_shape, type_json, filters=[deflate_filter])
            self.assertTrue(False)
        except TypeError:
            pass  # expected

        simple_shape = {"class": "H5S_SIMPLE", "dims": [10, 20]}
        plist = generate_dcpl(simple_shape, type_json)
        self.assertTrue("layout" in plist)
        self.assertEqual(plist["layout"]["class"], "H5D_CONTIGUOUS")
        self.assertTrue("filters" not in plist)

        plist = generate_dcpl(simple_shape, type_json, chunks=(5, 10))
        self.assertEqual(plist["layout"], {"class": "H5D_CHUNKED", "dims": [5, 10]})

        plist = generate_dcpl(simple_shape, type_json, filters=[deflate_filter])
        self.assertEqual(plist["filters"], [deflate_filter])


if __name__ == "__main__":
    # setup test files

    unittest.main()
