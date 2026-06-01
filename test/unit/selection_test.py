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

from h5json import selections
from h5json.selections import (
    H5S_SEL_POINTS,
    H5S_SEL_ALL,
    H5S_SEL_HYPERSLABS,
    H5S_SEL_FANCY,
    PointSelection,
    SimpleSelection,
    FancySelection,
    ScalarSelection,
)


def make_point_sel(shape, mask):
    """Build a PointSelection from a boolean ndarray mask."""
    sel = PointSelection(shape)
    sel[mask]
    return sel


class SimpleSelectionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SimpleSelectionTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testSelectAll(self):
        shape = (10,)
        sel = selections.select(shape, ...)
        self.assertIsInstance(sel, SimpleSelection)
        # __getitem__ always sets HYPERSLABS even for a full-range ellipsis
        self.assertEqual(sel.select_type, H5S_SEL_HYPERSLABS)
        self.assertEqual(sel.shape, shape)
        self.assertEqual(sel.nselect, 10)
        self.assertEqual(sel.shape, sel.mshape)

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (0,))
        self.assertEqual(bbox[1], shape)

    def testSelectAll2D(self):
        shape = (4, 5)
        sel = selections.select(shape, ...)
        self.assertIsInstance(sel, SimpleSelection)
        self.assertEqual(sel.select_type, H5S_SEL_HYPERSLABS)
        self.assertEqual(sel.nselect, 20)
        self.assertEqual(sel.shape, sel.mshape)

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (0, 0))
        self.assertEqual(bbox[1], shape)

    def testSlice1D(self):
        shape = (10,)
        sel = selections.select(shape, slice(2, 7))
        self.assertIsInstance(sel, SimpleSelection)
        self.assertEqual(sel.select_type, H5S_SEL_HYPERSLABS)
        self.assertEqual(sel.start, (2,))
        self.assertEqual(sel.count, (5,))
        self.assertEqual(sel.step, (1,))
        self.assertEqual(sel.nselect, 5)

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (2,))
        self.assertEqual(bbox[1], (7,))

    def testSliceWithStep(self):
        shape = (10,)
        sel = selections.select(shape, slice(0, 10, 2))
        self.assertIsInstance(sel, SimpleSelection)
        self.assertEqual(sel.select_type, H5S_SEL_HYPERSLABS)
        self.assertEqual(sel.start, (0,))
        self.assertEqual(sel.count, (5,))
        self.assertEqual(sel.step, (2,))
        self.assertEqual(sel.nselect, 5)

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (0,))
        self.assertEqual(bbox[1], (9,))

    def testSlice2D(self):
        shape = (8, 10)
        sel = selections.select(shape, (slice(1, 4), slice(2, 9)))
        self.assertIsInstance(sel, SimpleSelection)
        self.assertEqual(sel.select_type, H5S_SEL_HYPERSLABS)
        self.assertEqual(sel.start, (1, 2))
        self.assertEqual(sel.count, (3, 7))
        self.assertEqual(sel.step, (1, 1))
        self.assertEqual(sel.nselect, 21)

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (1, 2))
        self.assertEqual(bbox[1], (4, 9))

    def testBroadcast1D(self):
        shape = (10,)
        sel = selections.select(shape, ...)
        self.assertIsInstance(sel, SimpleSelection)

        it = sel.broadcast((1,))
        count = 0
        for x in it:
            # start
            self.assertTrue(x[0][0] >= 0 and x[0][0] < 10)
            # count
            self.assertEqual(x[1], (1,))
            # step
            self.assertEqual(x[2], (1,))
            # scalar
            self.assertEqual(x[3], (False,))
            count += 1
        self.assertEqual(count, 10)

    def testBroadcast2D(self):
        shape = (8, 10)
        sel = selections.select(shape, ...)
        self.assertIsInstance(sel, SimpleSelection)
        try:
            sel.broadcast(4, 5)
            self.assertTrue(False)
        except TypeError:
            pass
        it = sel.broadcast((1, 10))
        count = 0
        for x in it:
            # start
            self.assertTrue(x[0][0] >= 0 and x[0][0] < 8)
            self.assertEqual(x[0][1], 0)
            # count
            self.assertEqual(x[1], (1, 10))
            # step
            self.assertEqual(x[2], (1, 1))
            # scalar
            self.assertEqual(x[3], (False, False))
            count += 1
        self.assertEqual(count, 8)

    def testSlices(self):
        shape = (8, 10)
        sel = selections.select(shape, (slice(2, 5), slice(3, 7)))
        self.assertEqual(sel.slices, (slice(2, 5, 1), slice(3, 7, 1)))

    def testNselect(self):
        shape = (100,)
        sel = selections.select(shape, slice(0, 100))
        self.assertEqual(sel.nselect, 100)
        sel2 = selections.select(shape, slice(10, 20))
        self.assertEqual(sel2.nselect, 10)

    def testOutOfRangeRaises(self):
        shape = (10,)
        # integer index out of range raises IndexError; slices are silently clamped
        with self.assertRaises(IndexError):
            selections.select(shape, 15)

    def testGetQueryParam1D(self):
        shape = (10,)
        sel = selections.select(shape, slice(2, 8))
        param = sel.getQueryParam()
        self.assertEqual(param, "[2:8]")

    def testGetQueryParam2D(self):
        shape = (8, 10)
        sel = selections.select(shape, (slice(1, 4), slice(0, 10)))
        param = sel.getQueryParam()
        self.assertEqual(param, "[1:4,0:10]")

    def testRepr(self):
        shape = (10,)
        sel = selections.select(shape, slice(0, 5))
        self.assertIn("SimpleSelection", repr(sel))

    def testScalarDataset(self):
        # select() routes to ScalarSelection when obj has .shape == ()
        scalar_ds = np.array(42)
        sel = selections.select(scalar_ds, ...)
        self.assertIsInstance(sel, ScalarSelection)
        self.assertEqual(sel.select_type, H5S_SEL_ALL)
        self.assertEqual(sel.nselect, 1)


class PointSelectionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PointSelectionTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testBoolMask1D(self):
        shape = (10,)
        mask = np.zeros(10, dtype=bool)
        mask[[0, 3, 7]] = True
        sel = make_point_sel(shape, mask)
        self.assertIsInstance(sel, PointSelection)
        self.assertEqual(sel.select_type, H5S_SEL_POINTS)
        self.assertEqual(sel.nselect, 3)
        points = sel.points
        self.assertEqual(len(points), 3)
        for i in range(len(points)):
            pt = points[i]
            self.assertTrue(isinstance(pt, np.ndarray))
            self.assertEqual(pt.shape, (1,))
            self.assertTrue(pt[0] in (0, 3, 7))

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (0,))
        self.assertEqual(bbox[1], (8,))

    def testBoolMask2D(self):
        shape = (4, 5)
        mask = np.zeros(shape, dtype=bool)
        mask[0, 1] = True
        mask[2, 3] = True
        sel = make_point_sel(shape, mask)
        self.assertEqual(sel.select_type, H5S_SEL_POINTS)
        self.assertEqual(sel.nselect, 2)
        pts = sel.points
        self.assertEqual(pts.shape, (2, 2))
        self.assertEqual(list(pts[0]), [0, 1])
        self.assertEqual(list(pts[1]), [2, 3])

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (0, 1))
        self.assertEqual(bbox[1], (3, 4))

    def testListOfCoords1D(self):
        shape = (10,)
        sel = selections.select(shape, [2, 3, 5, 7])
        self.assertIsInstance(sel, PointSelection)
        self.assertEqual(sel.select_type, H5S_SEL_POINTS)
        self.assertEqual(sel.nselect, 4)
        points = sel.points
        self.assertEqual(len(points), 4)
        for i in range(len(points)):
            pt = points[i]
            self.assertTrue(pt in (2, 3, 5, 7))

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (2,))
        self.assertEqual(bbox[1], (8,))

    def testListOfCoords2D(self):
        shape = (8, 10)
        sel = selections.select(shape, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertIsInstance(sel, PointSelection)
        self.assertEqual(sel.select_type, H5S_SEL_POINTS)
        self.assertEqual(sel.nselect, 4)
        points = sel.points
        self.assertEqual(len(points), 4)
        for i in range(len(points)):
            pt = points[i]
            self.assertTrue(isinstance(pt, np.ndarray))
            self.assertEqual(pt.shape, (2,))
            self.assertTrue(pt[0] == pt[1])

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], (0, 0))
        self.assertEqual(bbox[1], (4, 4))

    def testEmptySet(self):
        shape = (10,)
        sel = PointSelection(shape)
        sel.set([])
        self.assertEqual(sel.nselect, 0)

        bbox = sel.bbox
        self.assertTrue(isinstance(bbox, tuple))
        self.assertEqual(len(bbox), 2)
        self.assertEqual(bbox[0], None)
        self.assertEqual(bbox[1], None)

    def testSetReplacesPoints(self):
        shape = (10,)
        mask1 = np.zeros(10, dtype=bool)
        mask1[[1, 2, 3]] = True
        sel = make_point_sel(shape, mask1)
        self.assertEqual(sel.nselect, 3)

        mask2 = np.zeros(10, dtype=bool)
        mask2[[5, 6]] = True
        sel[mask2]
        self.assertEqual(sel.nselect, 2)

    def testRepr(self):
        shape = (10,)
        mask = np.zeros(10, dtype=bool)
        mask[[0, 1]] = True
        sel = make_point_sel(shape, mask)
        self.assertIn("PointSelection", repr(sel))


class FancySelectionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FancySelectionTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testCoordList1D(self):
        shape = (10,)
        sel = FancySelection(shape)
        sel[[2, 5, 8]]
        self.assertEqual(sel.select_type, H5S_SEL_FANCY)

    def testGetQueryParamSlice(self):
        shape = (10,)
        sel = FancySelection(shape)
        sel[slice(2, 8)]
        param = sel.getQueryParam()
        self.assertEqual(param, "[2:8]")

    def testGetQueryParamList(self):
        shape = (10,)
        sel = FancySelection(shape)
        sel[[1, 3, 5]]
        param = sel.getQueryParam()
        self.assertEqual(param, "[[1,3,5]]")

    def testGetQueryParam2D(self):
        shape = (10, 10)
        sel = FancySelection(shape)
        sel[(slice(1, 4), slice(2, 6))]
        param = sel.getQueryParam()
        self.assertEqual(param, "[1:4,2:6]")

    def testRepr(self):
        shape = (10,)
        sel = FancySelection(shape)
        sel[slice(0, 5)]
        self.assertIn("FancySelection", repr(sel))


class IntersectHyperslabTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(IntersectHyperslabTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testOverlapping1D(self):
        shape = (10,)
        s1 = selections.select(shape, slice(0, 6))
        s2 = selections.select(shape, slice(3, 10))
        result = selections.intersect(s1, s2)
        self.assertIsInstance(result, SimpleSelection)
        self.assertEqual(result.nselect, 3)
        self.assertEqual(result.start, (3,))
        self.assertEqual(result.count, (3,))

    def testNonOverlapping1D(self):
        shape = (10,)
        s1 = selections.select(shape, slice(0, 3))
        s2 = selections.select(shape, slice(5, 10))
        result = selections.intersect(s1, s2)
        self.assertEqual(result.nselect, 0)

    def testOverlapping2D(self):
        shape = (10, 10)
        s1 = selections.select(shape, (slice(0, 6), slice(0, 6)))
        s2 = selections.select(shape, (slice(3, 10), slice(3, 10)))
        result = selections.intersect(s1, s2)
        self.assertEqual(result.nselect, 9)
        self.assertEqual(result.start, (3, 3))
        self.assertEqual(result.count, (3, 3))

    def testFullOverlap(self):
        shape = (10,)
        s1 = selections.select(shape, slice(2, 8))
        s2 = selections.select(shape, slice(0, 10))
        result = selections.intersect(s1, s2)
        self.assertEqual(result.nselect, 6)
        self.assertEqual(result.start, (2,))
        self.assertEqual(result.count, (6,))

    def testSelectAllWithHyperslab(self):
        shape = (10,)
        s_all = selections.select(shape, ...)
        s_hyp = selections.select(shape, slice(3, 7))
        result = selections.intersect(s_all, s_hyp)
        self.assertEqual(result.nselect, 4)
        self.assertEqual(result.start, (3,))

    def testSteppedSliceRaises(self):
        shape = (10,)
        s1 = selections.select(shape, slice(0, 10, 2))
        s2 = selections.select(shape, slice(0, 10, 2))
        with self.assertRaises(ValueError):
            selections.intersect(s1, s2)

    def testShapeMismatchRaises(self):
        s1 = selections.select((10,), slice(0, 5))
        s2 = selections.select((20,), slice(0, 5))
        with self.assertRaises(ValueError):
            selections.intersect(s1, s2)

    def testBadArgRaises(self):
        s1 = selections.select((10,), slice(0, 5))
        with self.assertRaises(TypeError):
            selections.intersect(s1, "not a selection")


class IntersectPointHyperslabTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(IntersectPointHyperslabTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testPointsInsideHyperslab1D(self):
        shape = (10,)
        mask = np.zeros(10, dtype=bool)
        mask[[0, 1, 3, 5, 9]] = True
        pts = make_point_sel(shape, mask)
        hyp = selections.select(shape, slice(2, 8))
        result = selections.intersect(pts, hyp)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 2)
        self.assertEqual(list(result.points.flatten()), [3, 5])

    def testHyperslabIntersectPoints1D(self):
        shape = (10,)
        mask = np.zeros(10, dtype=bool)
        mask[[0, 1, 3, 5, 9]] = True
        pts = make_point_sel(shape, mask)
        hyp = selections.select(shape, slice(2, 8))
        result = selections.intersect(hyp, pts)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 2)
        self.assertEqual(list(result.points.flatten()), [3, 5])

    def testAllPointsInsideHyperslab(self):
        shape = (10,)
        mask = np.zeros(10, dtype=bool)
        mask[[2, 4, 6]] = True
        pts = make_point_sel(shape, mask)
        hyp = selections.select(shape, slice(0, 10))
        result = selections.intersect(pts, hyp)
        self.assertEqual(result.nselect, 3)

    def testNoPointsInsideHyperslab(self):
        shape = (10,)
        mask = np.zeros(10, dtype=bool)
        mask[[0, 1]] = True
        pts = make_point_sel(shape, mask)
        hyp = selections.select(shape, slice(5, 10))
        result = selections.intersect(pts, hyp)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 0)

    def testPoints2DIntersectHyperslab(self):
        shape = (6, 6)
        pts = selections.select(shape, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        hyp = selections.select(shape, (slice(1, 4), slice(1, 4)))
        result = selections.intersect(pts, hyp)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 3)
        pts_list = [tuple(row) for row in result.points]
        self.assertIn((1, 1), pts_list)
        self.assertIn((2, 2), pts_list)
        self.assertIn((3, 3), pts_list)

    def testPoints2DIntersectSelectAll(self):
        shape = (5, 5)
        pts = selections.select(shape, [(0, 0), (2, 3), (4, 4)])
        s_all = selections.select(shape, ...)
        result = selections.intersect(pts, s_all)
        self.assertEqual(result.nselect, 3)

    def testHyperslabWithStep1D(self):
        shape = (20,)
        mask = np.zeros(20, dtype=bool)
        mask[[0, 2, 4, 6, 7]] = True
        pts = make_point_sel(shape, mask)
        # step-2 hyperslab covers 0,2,4,6,8,...
        hyp = selections.select(shape, slice(0, 10, 2))
        result = selections.intersect(pts, hyp)
        self.assertEqual(result.nselect, 4)
        self.assertEqual(list(result.points.flatten()), [0, 2, 4, 6])

    def testHyperslabFirstArg2D(self):
        # hyperslab as the first argument in 2-D
        shape = (8, 10)
        hyp = selections.select(shape, (slice(2, 6), slice(3, 8)))
        pts = selections.select(shape, [(1, 1), (2, 3), (3, 5), (5, 7), (6, 9)])
        result = selections.intersect(hyp, pts)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 3)
        pts_list = [tuple(row) for row in result.points]
        self.assertIn((2, 3), pts_list)
        self.assertIn((3, 5), pts_list)
        self.assertIn((5, 7), pts_list)

    def testDisjointBboxReturnsEmpty(self):
        # bounding boxes don't overlap at all — exercises the bbox fast path
        shape = (20,)
        mask = np.zeros(20, dtype=bool)
        mask[[0, 1, 2, 3, 4]] = True        # points in [0, 5)
        pts = make_point_sel(shape, mask)
        hyp = selections.select(shape, slice(10, 20))  # hyperslab in [10, 20)
        result = selections.intersect(hyp, pts)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 0)
        # commuted
        result2 = selections.intersect(pts, hyp)
        self.assertEqual(result2.nselect, 0)


class IntersectPointPointTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(IntersectPointPointTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testOverlapping1D(self):
        shape = (10,)
        mask1 = np.zeros(10, dtype=bool)
        mask1[[0, 1, 3, 5]] = True
        mask2 = np.zeros(10, dtype=bool)
        mask2[[1, 3, 7]] = True
        s1 = make_point_sel(shape, mask1)
        s2 = make_point_sel(shape, mask2)
        result = selections.intersect(s1, s2)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 2)
        self.assertEqual(list(result.points.flatten()), [1, 3])

    def testNoOverlap1D(self):
        shape = (10,)
        mask1 = np.zeros(10, dtype=bool)
        mask1[[0, 1]] = True
        mask2 = np.zeros(10, dtype=bool)
        mask2[[8, 9]] = True
        result = selections.intersect(make_point_sel(shape, mask1),
                                      make_point_sel(shape, mask2))
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 0)

    def testIdentical1D(self):
        shape = (10,)
        mask = np.zeros(10, dtype=bool)
        mask[[2, 5, 8]] = True
        s1 = make_point_sel(shape, mask)
        s2 = make_point_sel(shape, mask)
        result = selections.intersect(s1, s2)
        self.assertEqual(result.nselect, 3)
        self.assertEqual(list(result.points.flatten()), [2, 5, 8])

    def testOverlapping2D(self):
        shape = (6, 6)
        s1 = selections.select(shape, [(0, 0), (1, 1), (2, 2), (3, 3)])
        s2 = selections.select(shape, [(1, 1), (2, 2), (5, 5)])
        result = selections.intersect(s1, s2)
        self.assertIsInstance(result, PointSelection)
        self.assertEqual(result.nselect, 2)
        pts_list = [tuple(row) for row in result.points]
        self.assertIn((1, 1), pts_list)
        self.assertIn((2, 2), pts_list)

    def testNoOverlap2D(self):
        shape = (6, 6)
        s1 = selections.select(shape, [(0, 0), (1, 1)])
        s2 = selections.select(shape, [(3, 3), (4, 4)])
        result = selections.intersect(s1, s2)
        self.assertEqual(result.nselect, 0)

    def testCommutativity(self):
        shape = (10,)
        mask1 = np.zeros(10, dtype=bool)
        mask1[[0, 2, 4, 6]] = True
        mask2 = np.zeros(10, dtype=bool)
        mask2[[2, 4, 8]] = True
        s1 = make_point_sel(shape, mask1)
        s2 = make_point_sel(shape, mask2)
        r_fwd = selections.intersect(s1, s2)
        r_rev = selections.intersect(s2, s1)
        self.assertEqual(r_fwd.nselect, r_rev.nselect)
        self.assertEqual(list(r_fwd.points.flatten()), list(r_rev.points.flatten()))


class ContainedTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ContainedTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testContainedTrue(self):
        shape = (10,)
        s1 = selections.select(shape, slice(2, 5))
        s2 = selections.select(shape, slice(0, 10))
        self.assertTrue(selections.contained(s1, s2))

    def testContainedFalse(self):
        shape = (10,)
        s1 = selections.select(shape, slice(0, 6))
        s2 = selections.select(shape, slice(3, 10))
        self.assertFalse(selections.contained(s1, s2))

    def testContainedSelf(self):
        shape = (10,)
        s = selections.select(shape, slice(2, 8))
        self.assertTrue(selections.contained(s, s))

    def testContained2D(self):
        shape = (10, 10)
        inner = selections.select(shape, (slice(2, 5), slice(2, 5)))
        outer = selections.select(shape, (slice(0, 10), slice(0, 10)))
        self.assertTrue(selections.contained(inner, outer))
        self.assertFalse(selections.contained(outer, inner))


class TranslateTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TranslateTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testTranslate1D(self):
        shape = (10,)
        s1 = selections.select(shape, slice(2, 8))
        s2 = selections.select(shape, slice(4, 7))
        result = selections.translate(s1, s2)
        self.assertEqual(result.select_type, H5S_SEL_HYPERSLABS)
        self.assertEqual(result.start, (2,))
        self.assertEqual(result.count, (3,))

    def testTranslate2D(self):
        shape = (10, 10)
        s1 = selections.select(shape, (slice(2, 8), slice(2, 8)))
        s2 = selections.select(shape, (slice(4, 6), slice(4, 6)))
        result = selections.translate(s1, s2)
        self.assertEqual(result.select_type, H5S_SEL_HYPERSLABS)
        self.assertEqual(result.start, (2, 2))
        self.assertEqual(result.count, (2, 2))

    def testTranslate2DWithPoints(self):
        shape = (10, 10)
        s1 = selections.select(shape, (slice(2, 8), slice(2, 8)))
        s2 = selections.select(shape, [(2, 2), (3, 3), (9, 9)])

        result = selections.translate(s1, s2)
        self.assertEqual(result.select_type, H5S_SEL_POINTS)
        self.assertEqual(result.nselect, 2)

        self.assertEqual(result.points.shape, (2, 2))
        self.assertEqual(list(result.points[0]), [0, 0])
        self.assertEqual(list(result.points[1]), [1, 1])

    def testTranslateNoOverlapRaises(self):
        shape = (10,)
        s1 = selections.select(shape, slice(0, 3))
        s2 = selections.select(shape, slice(5, 8))
        with self.assertRaises(ValueError):
            selections.translate(s1, s2)


if __name__ == "__main__":
    unittest.main()
