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

from h5json.dset_util import guessChunk, shrinkChunk, getChunkSize, expandChunk
from h5json.dset_util import getDatasetLayoutClass, getContiguousLayout, getChunkDims
from h5json.dset_util import validateChunkLayout, getDatasetLayout


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
            validateChunkLayout(dset_json["shape"], type_json, layout)
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
            validateChunkLayout(dset_json["shape"], type_json, layout)
        except ValueError:
            self.assertTrue(False)  # should raise exception

        chunk_dims = getChunkDims(dset_json)
        self.assertEqual(chunk_dims, (2, ))

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

        shape = {"class": "H5S_SIMPLE", "dims": [100, 0], "maxdims": [100, 0]}
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
            "maxdims": [1000, 0, 1000],
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


if __name__ == "__main__":
    # setup test files

    unittest.main()
