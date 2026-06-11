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


"""
    High-level access to HDF5 dataspace selections
"""

from __future__ import absolute_import

import numpy as np


# Selection types
H5S_SEL_NONE = 0
H5S_SEL_POINTS = 1
H5S_SEL_HYPERSLABS = 2
H5S_SEL_ALL = 3
H5S_SEL_FANCY = 4


# Boolean selection operations
H5S_SELECT_SET = 1
H5S_SELECT_APPEND = 2
H5S_SELECT_PREPEND = 3
H5S_SELECT_OR = 4
H5S_SELECT_NONE = 5
H5S_SELECT_NOTB = 6


def select(obj, args):
    """ High-level routine to generate a selection from arbitrary arguments
    to selection initializer.  The arguments should be the following:

    obj
        Dataset object

    args
        Either a single argument or a tuple of arguments.  See below for
        supported classes of argument.

    Argument classes:

    Single Selection instance
        Returns the argument.

    numpy.ndarray
        Must be a boolean mask.  Returns a PointSelection instance.

    RegionReference
        Returns a Selection instance.

    Indices, slices, ellipses only
        Returns a SimpleSelection instance

    Indices, slices, ellipses, lists or boolean index arrays
        Returns a FancySelection instance.
    """
    if not isinstance(args, tuple):
        args = (args,)

    if hasattr(obj, "shape"):
        obj_shape = obj.shape
    elif isinstance(obj, tuple):
        obj_shape = obj
    else:
        raise TypeError("Object must be a dataset or a shape tuple")

    if len(obj_shape) == 0:
        # scalar object
        sel = ScalarSelection(obj_shape, args)
        return sel

    # "Special" indexing objects
    if len(args) == 1:

        arg = args[0]
        if hasattr(arg, "shape"):
            arg_shape = arg.shape
        else:
            arg_shape = obj_shape

        if isinstance(arg, Selection):
            if arg_shape != obj_shape:
                raise TypeError("Mismatched selection shape")
            return arg

        elif isinstance(arg, np.ndarray) or isinstance(arg, list):
            sel = PointSelection(obj_shape)
            # sel[arg]
            sel.set(arg)
            return sel
        """
        #todo - RegionReference
        elif isinstance(arg, h5r.RegionReference):
            sid = h5r.get_region(arg, dsid)
            if shape != sid.shape:
                raise TypeError("Reference shape does not match dataset shape")

            return Selection(shape, spaceid=sid)
        """

    for a in args:
        use_fancy = False
        if isinstance(a, np.ndarray):
            use_fancy = True
        elif a is []:
            use_fancy = True
        elif not isinstance(a, slice) and a is not Ellipsis:
            try:
                int(a)
            except Exception:
                use_fancy = True
        if use_fancy:
            sel = FancySelection(obj_shape, args)
            return sel

    sel = SimpleSelection(obj_shape, args)

    return sel


def _check_bool_args(s1, s2):
    """ verify argument for boolean operations """
    # TBD: this is currently only working for simple selections with stride 1
    valid_s1_types = (H5S_SEL_HYPERSLABS, H5S_SEL_ALL)
    valid_s2_types = (H5S_SEL_HYPERSLABS, H5S_SEL_POINTS, H5S_SEL_ALL)

    if not isinstance(s1, Selection):
        raise TypeError("Expected selection type for first arg")
    if not isinstance(s2, Selection):
        raise TypeError("Expected selection type for second arg")
    if s1.select_type not in valid_s1_types:
        raise TypeError("Expected hyperslab selection for first arg")
    if s2.select_type not in valid_s2_types:
        raise TypeError("Expected hyperslab selection for second arg")
    if s1.shape != s2.shape:
        raise ValueError("selections have incompatible shapes")


def _iter_points(point_sel):
    """Yield each point in a PointSelection as a tuple of ints."""
    pts = point_sel.points
    rank = len(point_sel.shape)
    pts_arr = np.asarray(pts)

    if pts_arr.size == 0:
        return

    if pts_arr.ndim == 1:
        if rank == 1:
            # Each scalar element is a coordinate in 1-D space
            for p in pts_arr:
                yield (int(p),)
        else:
            # Single point in rank-N space stored as a flat array [c0, c1, ..., c_{N-1}]
            yield tuple(int(x) for x in pts_arr)
    else:
        # Shape (N, rank): each row is one point
        for row in pts_arr:
            yield tuple(int(x) for x in row)


def _bboxes_overlap(s1, s2):
    """Return True if the bounding boxes of s1 and s2 overlap in every dimension."""
    min1, max1 = s1.bbox
    if min1 is None:
        return False
    min2, max2 = s2.bbox
    if min2 is None:
        return False
    return all(min1[d] < max2[d] and min2[d] < max1[d] for d in range(len(s1.shape)))


def _empty_point_sel(shape):
    """Return an empty PointSelection for the given shape."""
    result = PointSelection(shape)
    result.set([])
    return result


def _filter_points_by_hyperslab(point_sel, hyper_sel):
    """Return a PointSelection of points from point_sel that lie within hyper_sel."""
    if not _bboxes_overlap(point_sel, hyper_sel):
        return _empty_point_sel(point_sel.shape)

    start = hyper_sel.start
    count = hyper_sel.count
    step = hyper_sel.step
    rank = len(point_sel.shape)

    result_pts = []
    for pt in _iter_points(point_sel):
        if all(
            start[d] <= pt[d] < start[d] + count[d] * step[d] and (pt[d] - start[d]) % step[d] == 0
            for d in range(rank)
        ):
            result_pts.append(pt)

    result = PointSelection(point_sel.shape)
    if rank == 1:
        result.set([p[0] for p in result_pts] if result_pts else [])
    else:
        result.set(result_pts if result_pts else [])
    return result


def _intersect_points_points(s1, s2):
    """Return a PointSelection of points common to both s1 and s2."""
    if not _bboxes_overlap(s1, s2):
        return _empty_point_sel(s1.shape)

    common = sorted(set(_iter_points(s1)) & set(_iter_points(s2)))

    rank = len(s1.shape)
    result = PointSelection(s1.shape)
    if rank == 1:
        result.set([p[0] for p in common] if common else [])
    else:
        result.set(common if common else [])
    return result


def _intersect_fancy_hyperslab(fancy_sel, hyper_sel):
    """Return the intersection of a FancySelection with a hyperslab selection.

    For each dimension, slice ranges are clipped and coordinate lists are
    filtered to those that fall within the hyperslab.  Returns an empty
    PointSelection when the intersection is empty.
    """
    rank = len(fancy_sel.shape)
    h_start = hyper_sel.start
    h_count = hyper_sel.count
    h_step = hyper_sel.step

    new_slices = []
    for dim in range(rank):
        s = fancy_sel.slices[dim]
        hs = h_start[dim]
        hc = h_count[dim]
        hst = h_step[dim]

        if isinstance(s, slice):
            if s.step > 1 or hst > 1:
                raise ValueError("stepped slices not currently supported")
            new_start = max(s.start, hs)
            new_stop = min(s.stop, hs + hc)
            if new_stop <= new_start:
                return _empty_point_sel(fancy_sel.shape)
            new_slices.append(slice(new_start, new_stop, 1))
        elif isinstance(s, list):
            if hst == 1:
                filtered = [x for x in s if hs <= x < hs + hc]
            else:
                filtered = [x for x in s if hs <= x < hs + hc * hst and (x - hs) % hst == 0]
            if not filtered:
                return _empty_point_sel(fancy_sel.shape)
            new_slices.append(filtered)
        elif isinstance(s, int):
            if hst == 1:
                in_range = hs <= s < hs + hc
            else:
                in_range = hs <= s < hs + hc * hst and (s - hs) % hst == 0
            if not in_range:
                return _empty_point_sel(fancy_sel.shape)
            new_slices.append(s)
        else:
            raise TypeError(f"Unexpected FancySelection slice type: {type(s)}")

    return FancySelection(fancy_sel.shape, new_slices)


def intersect(s1, s2):
    """ Return the intersection of two selections.

    Supports hyperslab/hyperslab, hyperslab/point, point/point, and
    hyperslab/fancy combinations.
    """
    if not isinstance(s1, Selection):
        raise TypeError("Expected selection type for first arg")
    if not isinstance(s2, Selection):
        raise TypeError("Expected selection type for second arg")
    if s1.shape != s2.shape:
        raise ValueError("selections have incompatible shapes")

    t1 = s1.select_type
    t2 = s2.select_type
    hyperslab_types = (H5S_SEL_HYPERSLABS, H5S_SEL_ALL)

    if t1 in hyperslab_types and t2 in hyperslab_types:
        slices = []
        rank = len(s1.shape)
        for dim in range(rank):
            start = max(s1.start[dim], s2.start[dim])
            stop = min(s1.start[dim] + s1.count[dim], s2.start[dim] + s2.count[dim])
            if s1.step[dim] > 1 or s2.step[dim] > 1:
                raise ValueError("stepped slices not currently supported")
            if start > stop:
                stop = start
            slices.append(slice(start, stop, 1))
        return select(s1.shape, tuple(slices))

    if t1 == H5S_SEL_POINTS and t2 in hyperslab_types:
        return _filter_points_by_hyperslab(s1, s2)

    if t1 in hyperslab_types and t2 == H5S_SEL_POINTS:
        return _filter_points_by_hyperslab(s2, s1)

    if t1 == H5S_SEL_POINTS and t2 == H5S_SEL_POINTS:
        return _intersect_points_points(s1, s2)

    if t1 == H5S_SEL_FANCY and t2 in hyperslab_types:
        return _intersect_fancy_hyperslab(s1, s2)

    if t1 in hyperslab_types and t2 == H5S_SEL_FANCY:
        return _intersect_fancy_hyperslab(s2, s1)

    raise TypeError(f"Unsupported selection types for intersection: {t1}, {t2}")


def _dim_contained(s1_dim, s2_dim):
    """Return True if every value represented by s1_dim is also in s2_dim.

    Each argument is a per-dimension component: a slice, list of ints, or int.
    Stepped slices are handled conservatively (return False).
    """
    # Normalise s1 to either a contiguous range or an explicit set.
    if isinstance(s1_dim, int):
        s1_start, s1_stop = s1_dim, s1_dim + 1
        s1_contiguous = True
    elif isinstance(s1_dim, list):
        s1_set = set(s1_dim)
        s1_contiguous = False
    elif isinstance(s1_dim, slice):
        s1_start = s1_dim.start if s1_dim.start is not None else 0
        s1_stop = s1_dim.stop
        s1_step = s1_dim.step if s1_dim.step is not None else 1
        if s1_step > 1:
            return False  # conservative for stepped slices
        s1_contiguous = True
    else:
        return False

    if isinstance(s2_dim, slice):
        s2_start = s2_dim.start if s2_dim.start is not None else 0
        s2_stop = s2_dim.stop
        s2_step = s2_dim.step if s2_dim.step is not None else 1
        if s2_step > 1:
            return False
        if s1_contiguous:
            return s1_start >= s2_start and s1_stop <= s2_stop
        else:
            return all(s2_start <= x < s2_stop for x in s1_set)
    elif isinstance(s2_dim, list):
        s2_set = set(s2_dim)
        if s1_contiguous:
            return all(x in s2_set for x in range(s1_start, s1_stop))
        else:
            return s1_set <= s2_set
    elif isinstance(s2_dim, int):
        if s1_contiguous:
            return s1_start == s2_dim and s1_stop == s2_dim + 1
        else:
            return s1_set == {s2_dim}
    else:
        return False


def _fancy_contained(s1, s2):
    """Return True if every element selected by s1 is also selected by s2.

    At least one of s1/s2 must be a FancySelection; the other may be a
    SimpleSelection (hyperslab or select-all).

    FancySelections with multiple list dimensions represent paired (non-grid)
    coordinates.  Containment for those is returned False conservatively.
    """
    rank = len(s1.shape)
    hyperslab_types = (H5S_SEL_HYPERSLABS, H5S_SEL_ALL)

    def get_dims(sel):
        if sel.select_type in hyperslab_types:
            return [
                slice(sel.start[d], sel.start[d] + sel.count[d] * sel.step[d], sel.step[d])
                for d in range(rank)
            ]
        else:  # H5S_SEL_FANCY
            return list(sel.slices)

    s1_dims = get_dims(s1)
    s2_dims = get_dims(s2)

    # Paired-coordinate FancySelections (multiple list dims) are not a
    # Cartesian product — per-dimension checking would be incorrect.
    if sum(1 for d in s1_dims if isinstance(d, list)) > 1:
        return False

    return all(_dim_contained(s1_dims[d], s2_dims[d]) for d in range(rank))


def contained(s1, s2):
    """ return True if s1 is contained in s2, otherwise False """
    if not isinstance(s1, Selection):
        raise TypeError("Expected selection type for first arg")
    if not isinstance(s2, Selection):
        raise TypeError("Expected selection type for second arg")
    if s1.shape != s2.shape:
        raise ValueError("selections have incompatible shapes")

    t1 = s1.select_type
    t2 = s2.select_type
    fancy_types = (H5S_SEL_FANCY,)
    hyperslab_types = (H5S_SEL_HYPERSLABS, H5S_SEL_ALL)

    if t1 in fancy_types or t2 in fancy_types:
        allowed = hyperslab_types + fancy_types
        if t1 not in allowed:
            raise TypeError(f"Unsupported selection type for contained(): {t1}")
        if t2 not in allowed:
            raise TypeError(f"Unsupported selection type for contained(): {t2}")
        return _fancy_contained(s1, s2)

    _check_bool_args(s1, s2)

    is_contained = True
    rank = len(s1.shape)
    for dim in range(rank):
        if s1.step[dim] > 1 or s2.step[dim] > 1:
            # TBD: do the right thing for stepped selections
            # for now just return False
            is_contained = False
            break
        if s1.start[dim] < s2.start[dim]:
            is_contained = False
            break
        if s1.start[dim] + s1.count[dim] > s2.start[dim] + s2.count[dim]:
            is_contained = False
            break
    return is_contained


def translate(s1, s2):
    """ Given two selections, s1 and s2, return a new selection
    definied by s2 relative to s1's start and count.
    s2 must be contained in s1 """

    if s1.select_type == H5S_SEL_FANCY:
        if not isinstance(s2, Selection):
            raise TypeError("Expected selection type for second arg")
        if s2.select_type not in (H5S_SEL_HYPERSLABS, H5S_SEL_ALL):
            raise TypeError("translate with FancySelection s1 only supports hyperslab s2")
        if s1.shape != s2.shape:
            raise ValueError("selections have incompatible shapes")

        sel_inter = intersect(s1, s2)
        if sel_inter.nselect == 0:
            raise ValueError("translate - selections not overlapping")

        rank = len(s1.shape)
        new_slices = []
        for dim in range(rank):
            s1_dim = s1.slices[dim]
            inter_dim = sel_inter.slices[dim]
            if isinstance(s1_dim, slice):
                offset = s1_dim.start if s1_dim.start is not None else 0
                if isinstance(inter_dim, slice):
                    new_slices.append(slice(inter_dim.start - offset, inter_dim.stop - offset, inter_dim.step))
                elif isinstance(inter_dim, list):
                    new_slices.append([x - offset for x in inter_dim])
                else:
                    new_slices.append(inter_dim - offset)
            elif isinstance(s1_dim, list):
                index_map = {val: idx for idx, val in enumerate(s1_dim)}
                if isinstance(inter_dim, list):
                    new_slices.append([index_map[x] for x in inter_dim])
                else:
                    new_slices.append(index_map[inter_dim])
        return FancySelection(s1.shape, new_slices)

    if s2.select_type == H5S_SEL_FANCY:
        if not isinstance(s1, Selection):
            raise TypeError("Expected selection type for first arg")
        if s1.select_type not in (H5S_SEL_HYPERSLABS, H5S_SEL_ALL):
            raise TypeError("Expected hyperslab selection for first arg")
        if s1.shape != s2.shape:
            raise ValueError("selections have incompatible shapes")

        sel_inter = intersect(s1, s2)
        if sel_inter.nselect == 0:
            raise ValueError("translate - selections not overlapping")

        rank = len(s1.shape)
        new_slices = []
        for dim in range(rank):
            s = sel_inter.slices[dim]
            offset = s1.start[dim]
            if isinstance(s, slice):
                new_slices.append(slice(s.start - offset, s.stop - offset, s.step))
            elif isinstance(s, list):
                new_slices.append([x - offset for x in s])
            else:  # int
                new_slices.append(s - offset)
        return FancySelection(s1.shape, new_slices)

    _check_bool_args(s1, s2)
    sel_inter = intersect(s1, s2)
    if sel_inter.nselect == 0:
        raise ValueError("translate - selections not overlapping")

    rank = len(s1.shape)
    args = []
    if s2.select_type == H5S_SEL_POINTS:
        points = []
        for pt in _iter_points(sel_inter):
            for d in range(rank):
                if pt[d] < s1.start[d] or pt[d] >= s1.start[d] + s1.count[d]:
                    continue
            points.append(tuple(pt[d] - s1.start[d] for d in range(rank)))
        if len(points) == 0:
            raise ValueError("translate - selections not overlapping")
        args.append(points)
    elif s2.select_type == H5S_SEL_HYPERSLABS:
        for dim in range(rank):
            start = s2.start[dim] - s1.start[dim]
            count = s2.count[dim]
            args.append(slice(start, start + count, 1))
    else:
        raise TypeError("translate - unsupported selection type for s2")
    return select(s1.shape, tuple(args))


class Selection(object):

    """
        Base class for HDF5 dataspace selections.  Subclasses support the
        "selection protocol", which means they have at least the following
        members:

        __init__(shape)   => Create a new selection on "shape"-tuple
        __getitem__(args) => Perform a selection with the range specified.
                             What args are allowed depends on the
                             particular subclass in use.

        id (read-only) =>      h5py.h5s.SpaceID instance
        shape (read-only) =>   The shape of the dataspace.
        mshape  (read-only) => The shape of the selection region.
                               Not guaranteed to fit within "shape", although
                               the total number of points is less than
                               product(shape).
        nselect (read-only) => Number of selected points.  Always equal to
                               product(mshape).

        broadcast(target_shape) => Return an iterable which yields dataspaces
                                   for read, based on target_shape.

        The base class represents "unshaped" selections (1-D).
    """

    def __init__(self, shape):
        """ Create a selection.   """

        shape = tuple(shape)
        self._shape = shape

        self._select_type = H5S_SEL_ALL

    @property
    def select_type(self):
        """ SpaceID instance """
        return self._select_type

    @property
    def shape(self):
        """ Shape of whole dataspace """
        return self._shape

    @property
    def bbox(self):
        """ Bounding box of selection, as a tuple of (min, max) corner coordinates.

        For point-based selections, this is the smallest hyperslab that contains
        all selected points.  For hyperslab-based selections, this is the
        smallest hyperslab that contains the selection (which may be larger than
        the actual selection if stepped slices are used).
        """
        if self._select_type == H5S_SEL_POINTS:
            pts_arr = np.asarray(self._points)
            if pts_arr.size == 0:
                return None, None
            # For rank-1, pts_arr is 1-D (shape (N,)); reshape so axis=0 reduces over points.
            rank = len(self._shape)
            if pts_arr.ndim == 1 and rank == 1:
                pts_arr = pts_arr.reshape(-1, 1)
            min_corner = tuple(int(x) for x in np.min(pts_arr, axis=0))
            max_corner = tuple(int(x) + 1 for x in np.max(pts_arr, axis=0))
            return min_corner, max_corner
        elif self._select_type in (H5S_SEL_HYPERSLABS, H5S_SEL_ALL):
            start = self.start
            stop = tuple(start[dim] + (self.count[dim] - 1) * self.step[dim] + 1 for dim in range(len(self._shape)))
            return start, stop
        else:
            raise TypeError("Bounding box is not defined for this selection type")

    @property
    def nselect(self):
        """ Number of elements currently selected """

        return self.getSelectNpoints()

    @property
    def mshape(self):
        """ Shape of selection (always 1-D for this class) """
        return (self.nselect,)

    @property
    def tgtshape(self):
        """ shape of selection in rank of dataspace"""
        return self.mshape

    def getSelectNpoints(self):
        npoints = None
        if self._select_type == H5S_SEL_NONE:
            npoints = 0
        elif self._select_type == H5S_SEL_ALL:
            dims = self._shape
            npoints = 1
            for nextent in dims:
                npoints *= nextent
        else:
            raise IOError("Unsupported select type")
        return npoints

    def broadcast(self, target_shape):
        """ Get an iterable for broadcasting """
        if np.product(target_shape) != self.nselect:
            raise TypeError("Broadcasting is not supported for point-wise selections")
        yield self._id

    def __getitem__(self, args):
        raise NotImplementedError("This class does not support indexing")

    def __repr__(self):
        return f"Selection(shape:{self._shape})"


class PointSelection(Selection):

    """
        Represents a point-wise selection.  You can supply sequences of
        points to the three methods append(), prepend() and set(), or a
        single boolean array to __getitem__.
    """
    def __init__(self, shape, points=None):
        """ Create a Point selection.   """
        Selection.__init__(self, shape)
        self._points = np.empty((0,), dtype=np.uint64)
        self._select_type = H5S_SEL_POINTS
        if points is not None:
            self._perform_selection(points, H5S_SELECT_SET)

    @property
    def points(self):
        """ selection points """
        return self._points

    def getSelectNpoints(self):
        npoints = None
        if self._select_type == H5S_SEL_NONE:
            npoints = 0
        elif self._select_type == H5S_SEL_ALL:
            dims = self._shape
            npoints = 1
            for nextent in dims:
                npoints *= nextent
        elif self._select_type == H5S_SEL_POINTS:
            dims = self._shape
            rank = len(dims)
            if len(self._points) == rank and not type(self._points[0]) in (list, tuple, np.ndarray):
                npoints = 1
            else:
                npoints = len(self._points)
        else:
            raise IOError("Unsupported select type")
        return npoints

    def _perform_selection(self, points, op):
        """ Internal method which actually performs the selection """
        points = np.asarray(points, order='C', dtype='u8')

        if self._select_type != H5S_SEL_POINTS:
            op = H5S_SELECT_SET
        self._select_type = H5S_SEL_POINTS

        if op == H5S_SELECT_SET:
            self._points = points
        elif op == H5S_SELECT_APPEND:
            self._points.extent(points)
        elif op == H5S_SELECT_PREPEND:
            tmp = self._points
            self._points = points
            self._points.extend(tmp)
        else:
            raise ValueError("Unsupported operation")

    def append(self, points):
        """ Add the sequence of points to the end of the current selection """
        self._perform_selection(points, H5S_SELECT_APPEND)

    def prepend(self, points):
        """ Add the sequence of points to the beginning of the current selection """
        self._perform_selection(points, H5S_SELECT_PREPEND)

    def set(self, points):
        """ Replace the current selection with the given sequence of points"""

        if isinstance(points, np.ndarray) and points.dtype.kind == 'b':
            # boolean array selection
            if not points.shape == self._shape:
                raise TypeError("Boolean indexing array has incompatible shape")
            if not points.shape == self._shape:
                raise TypeError("Boolean indexing array has incompatible shape")
            self._perform_selection(points, H5S_SELECT_SET)

        elif isinstance(points, list) or isinstance(points, np.ndarray):
            # selection with list of points
            self._perform_selection(points, H5S_SELECT_SET)
        else:
            raise TypeError("PointSelection set() only works with list or numpy arrays")

    def __repr__(self):
        return f"PointSelection(shape:{self._shape}, {len(self._points)} points)"


class SimpleSelection(Selection):

    """ A single "rectangular" (regular) selection composed of only slices
        and integer arguments.  Can participate in broadcasting.
    """

    @property
    def mshape(self):
        """ Shape of current selection """
        return self._mshape

    @property
    def tgtshape(self):
        """ shape of selection in rank of dataspace"""
        return [self.count[dim] for dim in range(len(self._shape))]

    @property
    def start(self):
        return self._sel[0]

    @property
    def count(self):
        return self._sel[1]

    @property
    def step(self):
        return self._sel[2]

    def __init__(self, shape, hyperslab=None):
        Selection.__init__(self, shape)
        rank = len(self._shape)
        if self._shape == ():
            if hyperslab is not None and hyperslab not in (Ellipsis, ()):
                raise TypeError("Invalid index for scalar dataset (only ..., () allowed)")
            self._select_type = H5S_SEL_ALL
            self._mshape = ()
            return self

        if hyperslab is None:
            self._sel = ((0,) * rank, self._shape, (1,) * rank, (False,) * rank)
            self._mshape = self._shape
            self._select_type = H5S_SEL_ALL
        else:
            self._sel = _handle_simple(self._shape, hyperslab)
            self._mshape = tuple(x for x, y in zip(self._sel[1], self._sel[3]) if not y)
            self._select_type = H5S_SEL_HYPERSLABS

    def getSelectNpoints(self):
        """Return number of elements in current selection
        """
        npoints = None
        if self._select_type == H5S_SEL_NONE:
            npoints = 0
        elif self._select_type == H5S_SEL_ALL:
            dims = self._shape
            npoints = 1
            for nextent in dims:
                npoints *= nextent
        elif self._select_type == H5S_SEL_HYPERSLABS:
            dims = self._shape
            npoints = 1
            rank = len(dims)
            for i in range(rank):
                npoints *= self.count[i]
        else:
            raise IOError("Unsupported select type")
        return npoints

    def getQueryParam(self):
        """ Get select param for use with HDF Rest API"""
        param = ''
        rank = len(self._shape)
        if rank == 0:
            return None

        param += "["
        for i in range(rank):
            start = self.start[i]
            stop = start + (self.count[i] * self.step[i])
            if stop > self._shape[i]:
                stop = self._shape[i]
            dim_sel = str(start) + ':' + str(stop)
            if self.step[i] != 1:
                dim_sel += ':' + str(self.step[i])
            if i != rank - 1:
                dim_sel += ','
            param += dim_sel
        param += ']'
        return param

    def broadcast(self, target_shape):
        """ Return an iterator over target dataspaces for broadcasting.

        Follows the standard NumPy broadcasting rules against the current
        selection shape (self._mshape).
        """
        if self._shape == ():
            if np.product(target_shape) != 1:
                raise TypeError(f"Can't broadcast {target_shape} to scalar")
            yield self._sel
            return

        start, count, step, scalar = self._sel

        rank = len(count)
        target = list(target_shape)

        tshape = []
        for idx in range(1, rank + 1):
            if len(target) == 0 or scalar[-idx]:     # Skip scalar axes
                tshape.append(1)
            else:
                t = target.pop()
                if t == 1 or count[-idx] == t:
                    tshape.append(t)
                else:
                    raise TypeError(f"Can't broadcast {target_shape} -> {count}")
        tshape.reverse()
        tshape = tuple(tshape)

        chunks = tuple(x // y for x, y in zip(count, tshape))
        nchunks = int(np.prod(chunks))

        if nchunks == 1:
            yield self._sel
        else:
            for idx in range(nchunks):
                offset = []
                for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start):
                    offset.append(int(x * y * z + s))
                offset = tuple(offset)
                sel = [tuple([sum(x) for x in zip(offset, start)]), tshape, step, scalar]
                yield sel

    @property
    def slices(self):
        """ return tuple of slices for this selection """
        rank = len(self.shape)
        slices = []
        for dim in range(rank):
            start = self.start[dim]
            stop = start + self.count[dim]
            step = self.step[dim]
            slices.append(slice(start, stop, step))
        return tuple(slices)

    def __repr__(self):
        s = f"SimpleSelection(shape:{self._shape}, start: {self._sel[0]},"
        s += f" count: {self._sel[1]}, step: {self._sel[2]}"
        return s


class FancySelection(Selection):

    """
        Implements advanced NumPy-style selection operations in addition to
        the standard slice-and-int behavior.

        Indexing arguments may be ints, slices, lists of indicies, or
        per-axis (1D) boolean arrays.

        Broadcasting is not supported for these selections.
    """

    @property
    def slices(self):
        return self._slices

    @property
    def mshape(self):
        """ Shape of current selection """
        return self._mshape

    def __init__(self, shape, coords=None):
        Selection.__init__(self, shape)
        rank = len(self._shape)
        if rank < 2:
            raise TypeError("FancySelection is only supported for rank 2 or higher")

        if coords is None:
            self._sel = ((0,) * rank, self._shape, (1,) * rank, (False,) * rank)
            self._mshape = self._shape
            self._select_type = H5S_SEL_ALL
            return self

        if len(coords) != rank:
            raise TypeError("Number of coordinate sets does not match dataset rank")

        select_type = H5S_SEL_HYPERSLABS  # will adjust if we have a coord

        # Create list of slices and/or coordinates
        slices = []
        mshape = []
        num_coordinates = None
        for idx in range(rank):
            length = self._shape[idx]
            arg = coords[idx]
            if isinstance(arg, slice):
                _, count, _ = _translate_slice(arg, length)  # raise exception for invalid slice
                if arg.start is None:
                    start = 0
                else:
                    start = arg.start
                if arg.stop is None:
                    stop = length
                else:
                    stop = arg.stop
                if arg.step is None:
                    step = 1
                else:
                    step = arg.step
                slices.append(slice(start, stop, step))
                mshape.append(count)

            elif hasattr(arg, 'dtype') and arg.dtype == np.dtype('bool'):
                if len(arg.shape) != 1:
                    raise TypeError("Boolean indexing arrays must be 1-D")
                arg = arg.nonzero()[0]
                try:
                    slices.append(list(arg))
                except TypeError:
                    pass
                else:
                    if sorted(arg) != list(arg):
                        raise TypeError("Indexing elements must be in increasing order")
                mshape.append(len(arg))
                select_type = H5S_SEL_FANCY
            elif isinstance(arg, list) or hasattr(arg, 'dtype'):
                # coordinate selection
                slices.append(arg)
                for x in arg:
                    if x < 0 or x >= length:
                        raise IndexError(f"Index ({arg}) out of range (0-{length - 1})")
                if num_coordinates is None:
                    num_coordinates = len(arg)
                elif num_coordinates == len(arg):
                    # second set of coordinates doesn't effect mshape
                    continue
                else:
                    # this shouldn't happen since HSDS would have thrown an error
                    raise ValueError("coordinate num element missmatch")
                mshape.append(len(arg))
                select_type = H5S_SEL_FANCY
            elif isinstance(arg, int):
                if arg < 0 or arg >= length:
                    raise IndexError(f"Index ({arg}) out of range (0-{length - 1})")
                slices.append(arg)
            elif isinstance(arg, type(Ellipsis)):
                slices.append(slice(0, length, 1))
            else:
                raise TypeError(f"Unexpected arg type: {arg} - {type(arg)}")
        self._slices = slices
        self._select_type = select_type
        self._mshape = tuple(mshape)

    def getSelectNpoints(self):
        """Return number of elements in current selection
        """
        npoints = 1
        for idx, s in enumerate(self._slices):
            if isinstance(s, slice):
                length = self._shape[idx]
                _, count, _ = _translate_slice(s, length)
            elif isinstance(s, list):
                count = len(s)
            else:
                # scalar selection
                count = 1
            npoints *= count

        return npoints

    def getQueryParam(self):
        """ Get select param for use with HDF Rest API"""
        query = []
        query.append('[')
        rank = len(self._slices)
        for dim, s in enumerate(self._slices):
            if isinstance(s, slice):
                if s.start is None and s.stop is None:
                    query.append(':')
                elif s.stop is None:
                    query.append(f"{s.start}:")
                else:
                    query.append(f"{s.start}:{s.stop}")
                if s.step and s.step != 1:
                    query.append(f":{s.step}")
            elif isinstance(s, list) or hasattr(s, 'dtype'):
                query.append('[')
                for idx, n in enumerate(s):
                    query.append(str(n))
                    if idx + 1 < len(s):
                        query.append(',')
                query.append(']')
            else:
                # scalar selection
                query.append(str(s))
            if dim + 1 < rank:
                query.append(',')
        query.append(']')
        return "".join(query)

    def broadcast(self, target_shape):
        raise TypeError("Broadcasting is not supported for complex selections")

    def __repr__(self):
        return f"FancySelection(shape:{self._shape}, slices: {self._slices})"


def _expand_ellipsis(args, rank):
    """ Expand ellipsis objects and fill in missing axes.
    """
    n_el = sum(1 for arg in args if arg is Ellipsis)
    if n_el > 1:
        raise ValueError("Only one ellipsis may be used.")
    elif n_el == 0 and len(args) != rank:
        args = args + (Ellipsis,)

    final_args = []
    n_args = len(args)
    for arg in args:

        if arg is Ellipsis:
            final_args.extend((slice(None, None, None),) * (rank - n_args + 1))
        else:
            final_args.append(arg)

    if len(final_args) > rank:
        raise TypeError("Argument sequence too long")

    return final_args


def _handle_simple(shape, args):
    """ Process a "simple" selection tuple, containing only slices and
        integer objects.  Return is a 4-tuple with tuples for start,
        count, step, and a flag which tells if the axis is a "scalar"
        selection (indexed by an integer).

        If "args" is shorter than "shape", the remaining axes are fully
        selected.
    """
    args = _expand_ellipsis(args, len(shape))

    start = []
    count = []
    step = []
    scalar = []

    for arg, length in zip(args, shape):
        if isinstance(arg, slice):
            x, y, z = _translate_slice(arg, length)
            s = False
        else:
            try:
                x, y, z = _translate_int(int(arg), length)
                s = True
            except TypeError:
                raise TypeError(f'Illegal index "{arg}" (must be a slice or number)')
        start.append(x)
        count.append(y)
        step.append(z)
        scalar.append(s)

    return tuple(start), tuple(count), tuple(step), tuple(scalar)


def _translate_int(exp, length):
    """ Given an integer index, return a 3-tuple
        (start, count, step)
        for hyperslab selection
    """
    if exp < 0:
        exp = length + exp

    if not 0 <= exp < length:
        raise IndexError(f"Index ({exp}) out of range (0-{length - 1})")

    return exp, 1, 1


def _translate_slice(exp, length):
    """ Given a slice object, return a 3-tuple
        (start, count, step)
        for use with the hyperslab selection routines
    """
    start, stop, step = exp.indices(length)
    # Now if step > 0, then start and stop are in [0, length];
    # if step < 0, they are in [-1, length - 1] (Python 2.6b2 and later;
    # Python issue 3004).

    if step < 1:
        raise ValueError("Step must be >= 1 (got %d)" % step)
    if stop < start:
        stop = start

    count = 1 + (stop - start - 1) // step

    return start, count, step


def guess_shape(sid):
    """ Given a dataspace, try to deduce the shape of the selection.

    Returns one of:
        * A tuple with the selection shape, same length as the dataspace
        * A 1D selection shape for point-based and multiple-hyperslab selections
        * None, for unselected scalars and for NULL dataspaces
    """

    sel_class = sid.get_simple_extent_type()    # Dataspace class
    sel_type = sid.get_select_type()            # Flavor of selection in use

    if sel_class == 'H5S_NULL':
        # NULL dataspaces don't support selections
        return None

    elif sel_class == 'H5S_SCALAR':
        # NumPy has no way of expressing empty 0-rank selections, so we use None
        if sel_type == H5S_SEL_NONE:
            return None
        if sel_type == H5S_SEL_ALL:
            return tuple()

    elif sel_class != 'H5S_SIMPLE':
        raise TypeError(f"Unrecognized dataspace class {sel_class}")

    # We have a "simple" (rank >= 1) dataspace

    N = sid.get_select_npoints()
    rank = len(sid.shape)

    if sel_type == H5S_SEL_NONE:
        return (0,) * rank

    elif sel_type == H5S_SEL_ALL:
        return sid.shape

    elif sel_type == H5S_SEL_POINTS:
        # Like NumPy, point-based selections yield 1D arrays regardless of
        # the dataspace rank
        return (N,)

    elif sel_type != H5S_SEL_HYPERSLABS:
        raise TypeError(f"Unrecognized selection method {sel_type}")

    # We have a hyperslab-based selection

    if N == 0:
        return (0,) * rank

    bottomcorner, topcorner = (np.array(x) for x in sid.get_select_bounds())

    # Shape of full selection box
    boxshape = topcorner - bottomcorner + np.ones((rank,))

    def get_n_axis(sid, axis):
        """ Determine the number of elements selected along a particular axis.

        To do this, we "mask off" the axis by making a hyperslab selection
        which leaves only the first point along the axis.  For a 2D dataset
        with selection box shape (X, Y), for axis 1, this would leave a
        selection of shape (X, 1).  We count the number of points N_leftover
        remaining in the selection and compute the axis selection length by
        N_axis = N/N_leftover.
        """

        if (boxshape[axis]) == 1:
            return 1

        start = bottomcorner.copy()
        start[axis] += 1
        count = boxshape.copy()
        count[axis] -= 1

        # Throw away all points along this axis
        masked_sid = sid.copy()
        masked_sid.select_hyperslab(tuple(start), tuple(count), op=H5S_SELECT_NOTB)

        N_leftover = masked_sid.get_select_npoints()

        return N // N_leftover

    shape = tuple(get_n_axis(sid, x) for x in range(rank))

    if np.product(shape) != N:
        # This means multiple hyperslab selections are in effect,
        # so we fall back to a 1D shape
        return (N,)

    return shape


class ScalarSelection(Selection):

    """
        Implements slicing for scalar datasets.
    """

    @property
    def mshape(self):
        return self._mshape

    def __init__(self, shape, *args, **kwds):
        Selection.__init__(self, shape)
        arg = None
        if len(args) > 0:
            arg = args[0]
        if arg == ():
            self._mshape = None
            self._select_type = H5S_SEL_ALL
        elif arg == (Ellipsis,):
            self._mshape = ()
            self._select_type = H5S_SEL_ALL
        else:
            raise ValueError("Illegal slicing argument for scalar dataspace")
