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


def select(obj, args, fields=None):
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
        Returns a SimpleSelection instance with H5S_SEL_HYPERSLABS.

    Indices, slices, ellipses, lists or boolean index arrays
        Returns a SimpleSelection instance with H5S_SEL_FANCY.
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
            return SimpleSelection(obj_shape, _points_to_paired(obj_shape, arg), fields=fields)
        """
        #todo - RegionReference
        elif isinstance(arg, h5r.RegionReference):
            sid = h5r.get_region(arg, dsid)
            if shape != sid.shape:
                raise TypeError("Reference shape does not match dataset shape")

            return Selection(shape, spaceid=sid)
        """

    sel = SimpleSelection(obj_shape, args, fields=fields)
    return sel


def _check_bool_args(s1, s2):
    """ verify argument for boolean operations """
    # TBD: this is currently only working for simple selections with stride 1
    valid_s1_types = (H5S_SEL_HYPERSLABS, H5S_SEL_POINTS, H5S_SEL_ALL)
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


def _points_to_paired(shape, points):
    """Convert a list of point coordinates or a boolean array into the
    per-dimension tuple expected by SimpleSelection's fancy path.

    Examples
    --------
    1-D shape (10,), scalar indices [3, 5, 7]  ->  ([3, 5, 7],)
    2-D shape (10,10), tuples [(1,2),(3,4)]    ->  ([1, 3], [2, 4])
    2-D boolean mask                            ->  (row_list, col_list)
    """
    rank = len(shape)
    arr = np.asarray(points)

    if arr.dtype.kind == 'b':
        # Boolean mask: nonzero() returns one index array per dimension.
        coords = arr.nonzero()
        return tuple(list(c.astype(int)) for c in coords)

    if arr.size == 0:
        return tuple([] for _ in range(rank))

    if arr.ndim == 1:
        if rank == 1:
            return ([int(x) for x in arr],)
        if len(arr) == rank:
            # Single point stored as a flat 1-D array [c0, c1, ..., c_{rank-1}]
            return tuple([int(arr[d])] for d in range(rank))
        raise TypeError(f"Cannot interpret 1-D array of length {len(arr)} as points for shape {shape}")

    if arr.ndim == 2 and arr.shape[1] == rank:
        # N×rank array: transpose to rank×N
        return tuple([int(arr[i, d]) for i in range(arr.shape[0])] for d in range(rank))

    raise TypeError(f"Cannot interpret array of shape {arr.shape} as point selection for shape {shape}")


def _iter_points(sel):
    """Yield each point in a paired-coordinate FancySelection as a tuple of ints."""
    slices = sel.slices
    rank = len(sel.shape)
    list_dims = [d for d in range(rank) if isinstance(slices[d], list)]
    if not list_dims:
        return
    n = len(slices[list_dims[0]])
    for i in range(n):
        yield tuple(int(slices[d][i]) for d in range(rank))


def _bboxes_overlap(s1, s2):
    """Return True if the bounding boxes of s1 and s2 overlap in every dimension."""
    min1, max1 = s1.bbox
    if min1 is None:
        return False
    min2, max2 = s2.bbox
    if min2 is None:
        return False
    return all(min1[d] < max2[d] and min2[d] < max1[d] for d in range(len(s1.shape)))


def _empty_paired_sel(shape):
    """Return an empty paired-coordinate FancySelection for the given shape."""
    rank = len(shape)
    return SimpleSelection(shape, tuple([] for _ in range(rank)))


def _intersect_paired_fancy(s1, s2):
    """Return the intersection of two paired-coordinate FancySelections."""
    if not _bboxes_overlap(s1, s2):
        return _empty_paired_sel(s1.shape)

    rank = len(s1.shape)
    pts1 = set(zip(*[s1.slices[d] for d in range(rank)]))
    pts2 = set(zip(*[s2.slices[d] for d in range(rank)]))
    common = sorted(pts1 & pts2)

    if not common:
        return _empty_paired_sel(s1.shape)

    return SimpleSelection(s1.shape,
                           tuple([int(pt[d]) for pt in common] for d in range(rank)))


def _pt_in_hyperslab(val, hs, hc, hst):
    """Return True if scalar val lies within the hyperslab range for one dim."""
    if hst == 1:
        return hs <= val < hs + hc
    return hs <= val < hs + hc * hst and (val - hs) % hst == 0


def _intersect_fancy_hyperslab(fancy_sel, hyper_sel):
    """Return the intersection of a FancySelection with a hyperslab selection.

    For Cartesian-product selections (at most one list dimension) each
    dimension is clipped independently.  For paired-coordinate selections
    (multiple list dimensions) the coordinate pairs are filtered as a unit so
    the two lists always retain the same length.  Returns an empty
    paired FancySelection when the intersection is empty.
    """
    rank = len(fancy_sel.shape)
    h_start = hyper_sel.start
    h_count = hyper_sel.count
    h_step = hyper_sel.step
    slices = fancy_sel.slices  # tuple after the property fix

    list_dims = [d for d in range(rank) if isinstance(slices[d], list)]

    if len(list_dims) > 1:
        # Paired-coordinate selection: check slice dims first, then filter pairs.
        for dim in range(rank):
            s = slices[dim]
            hs, hc, hst = h_start[dim], h_count[dim], h_step[dim]
            if isinstance(s, slice):
                if s.step > 1 or hst > 1:
                    raise ValueError("stepped slices not currently supported")
                if min(s.stop, hs + hc) <= max(s.start, hs):
                    return _empty_paired_sel(fancy_sel.shape)
            elif isinstance(s, int):
                if not _pt_in_hyperslab(s, hs, hc, hst):
                    return _empty_paired_sel(fancy_sel.shape)

        n_pairs = len(slices[list_dims[0]])
        keep = [
            i for i in range(n_pairs)
            if all(_pt_in_hyperslab(slices[d][i], h_start[d], h_count[d], h_step[d])
                   for d in list_dims)
        ]
        if not keep:
            return _empty_paired_sel(fancy_sel.shape)

        new_slices = []
        for dim in range(rank):
            s = slices[dim]
            if isinstance(s, list):
                new_slices.append([s[i] for i in keep])
            elif isinstance(s, slice):
                hs, hc = h_start[dim], h_count[dim]
                new_slices.append(slice(max(s.start, hs), min(s.stop, hs + hc), 1))
            else:  # int: already validated above, keep as-is
                new_slices.append(s)
        return FancySelection(fancy_sel.shape, new_slices)

    # Cartesian-product path: clip each dimension independently.
    new_slices = []
    for dim in range(rank):
        s = slices[dim]
        hs = h_start[dim]
        hc = h_count[dim]
        hst = h_step[dim]

        if isinstance(s, slice):
            if s.step > 1 or hst > 1:
                raise ValueError("stepped slices not currently supported")
            new_start = max(s.start, hs)
            new_stop = min(s.stop, hs + hc)
            if new_stop <= new_start:
                return _empty_paired_sel(fancy_sel.shape)
            new_slices.append(slice(new_start, new_stop, 1))
        elif isinstance(s, list):
            if hst == 1:
                filtered = [x for x in s if hs <= x < hs + hc]
            else:
                filtered = [x for x in s if hs <= x < hs + hc * hst and (x - hs) % hst == 0]
            if not filtered:
                return _empty_paired_sel(fancy_sel.shape)
            new_slices.append(filtered)
        elif isinstance(s, int):
            if not _pt_in_hyperslab(s, hs, hc, hst):
                return _empty_paired_sel(fancy_sel.shape)
            new_slices.append(s)
        else:
            raise TypeError(f"Unexpected FancySelection slice type: {type(s)}")

    return FancySelection(fancy_sel.shape, new_slices)


def _intersect_fields(f1, f2):
    """Return the fields for an intersection result.

    None means 'all fields'.  The intersection of two field sets is the set of
    fields that appear in both; intersecting with None (all fields) yields the
    other operand's fields unchanged.
    """
    if f1 is None:
        return f2
    if f2 is None:
        return f1
    return f1 & f2  # both are sets


def intersect(s1, s2):
    """ Return the intersection of two selections.

    Supports hyperslab/hyperslab, hyperslab/fancy, and paired-fancy/fancy
    combinations.  The fields of the result are the intersection of the fields
    of the two input selections (None meaning 'all fields').
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
    result_fields = _intersect_fields(s1.fields, s2.fields)

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
        result = select(s1.shape, tuple(slices))
        result._fields = result_fields
        return result

    if t1 in (H5S_SEL_FANCY, H5S_SEL_POINTS) and t2 in hyperslab_types:
        result = _intersect_fancy_hyperslab(s1, s2)
        result._fields = result_fields
        return result

    if t1 in hyperslab_types and t2 in (H5S_SEL_FANCY, H5S_SEL_POINTS):
        result = _intersect_fancy_hyperslab(s2, s1)
        result._fields = result_fields
        return result

    if t1 == H5S_SEL_POINTS and t2 == H5S_SEL_POINTS:
        rank = len(s1.shape)
        # Only paired-coordinate (all-list-dim) selections are supported.
        s1_all_lists = sum(1 for s in s1.slices if isinstance(s, list)) == rank
        s2_all_lists = sum(1 for s in s2.slices if isinstance(s, list)) == rank
        if s1_all_lists and s2_all_lists:
            result = _intersect_paired_fancy(s1, s2)
            result._fields = result_fields
            return result
        raise TypeError(f"Unsupported selection types for intersection: {t1}, {t2}")

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


def _fields_contained(f1, f2):
    """Return True if the fields of s1 (f1) are contained in the fields of s2 (f2).

    None means 'all fields'.  s1 is field-contained in s2 when every field
    that s2 requests is also present in s1 — i.e. s2.fields ⊆ s1.fields.
    """
    if f2 is None:
        # s2 requests all fields; s1 must also cover all fields
        return f1 is None
    if f1 is None:
        # s1 has all fields, so any subset s2 requests is covered
        return True
    return f2 <= f1


def contained(s1, s2):
    """ return True if s1 is contained in s2, otherwise False.

    Takes compound-type fields into account: s1 is contained in s2 only if
    s2's field set is a subset of s1's field set (None means 'all fields').
    """
    if not isinstance(s1, Selection):
        raise TypeError("Expected selection type for first arg")
    if not isinstance(s2, Selection):
        raise TypeError("Expected selection type for second arg")
    if s1.shape != s2.shape:
        raise ValueError("selections have incompatible shapes")

    if not _fields_contained(s1.fields, s2.fields):
        return False

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


def _fancy_dim_intersect(s1_dim, s2_dim):
    """Return the per-dimension intersection of s1_dim and s2_dim in absolute
    coordinates, or None if the intersection is empty.

    Each argument is a per-dimension component: slice, list of ints, or int.
    For list×list, the result preserves s2's order.
    For list×slice, the result preserves s1's order (natural buffer order).
    """
    if isinstance(s1_dim, slice):
        s1_start, s1_stop = s1_dim.start, s1_dim.stop
        if isinstance(s2_dim, slice):
            start = max(s1_start, s2_dim.start)
            stop = min(s1_stop, s2_dim.stop)
            return slice(start, stop, 1) if stop > start else None
        elif isinstance(s2_dim, list):
            filtered = [x for x in s2_dim if s1_start <= x < s1_stop]
            return filtered if filtered else None
        else:  # int
            return s2_dim if s1_start <= s2_dim < s1_stop else None
    elif isinstance(s1_dim, list):
        s1_set = set(s1_dim)
        if isinstance(s2_dim, slice):
            filtered = [x for x in s1_dim if s2_dim.start <= x < s2_dim.stop]
            return filtered if filtered else None
        elif isinstance(s2_dim, list):
            filtered = [x for x in s2_dim if x in s1_set]
            return filtered if filtered else None
        else:  # int
            return s2_dim if s2_dim in s1_set else None
    elif isinstance(s1_dim, int):
        if isinstance(s2_dim, slice):
            return s1_dim if s2_dim.start <= s1_dim < s2_dim.stop else None
        elif isinstance(s2_dim, list):
            return s1_dim if s1_dim in s2_dim else None
        else:  # int
            return s1_dim if s1_dim == s2_dim else None
    return None


def translate(s1, s2):
    """ Given two selections, s1 and s2, return a new selection
    definied by s2 relative to s1's start and count.
    s2 must be contained in s1 """

    if s1.select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
        if not isinstance(s2, Selection):
            raise TypeError("Expected selection type for second arg")
        hyperslab_types = (H5S_SEL_HYPERSLABS, H5S_SEL_ALL)
        if s2.select_type not in (*hyperslab_types, H5S_SEL_FANCY, H5S_SEL_POINTS):
            raise TypeError(f"translate with FancySelection s1 does not support s2 type: {s2.select_type}")
        if s1.shape != s2.shape:
            raise ValueError("selections have incompatible shapes")

        rank = len(s1.shape)

        # Compute the intersection in absolute coordinates.
        if s2.select_type in hyperslab_types:
            sel_inter = intersect(s1, s2)
            if sel_inter.nselect == 0:
                raise ValueError("translate - selections not overlapping")
            inter_slices = sel_inter.slices
        else:  # s2 is also FancySelection
            inter_slices = []
            for dim in range(rank):
                inter_dim = _fancy_dim_intersect(s1.slices[dim], s2.slices[dim])
                if inter_dim is None:
                    raise ValueError("translate - selections not overlapping")
                inter_slices.append(inter_dim)

        # Express the intersection in s1's local coordinate frame.
        new_slices = []
        for dim in range(rank):
            s1_dim = s1.slices[dim]
            inter_dim = inter_slices[dim]
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
            else:  # int: scalar-indexed dim, local coordinate is always 0
                new_slices.append(inter_dim - s1_dim)
        return FancySelection(s1.shape, new_slices)

    if s2.select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
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
    if s2.select_type == H5S_SEL_HYPERSLABS:
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

        shape (read-only) =>   The shape of the dataspace.
        mshape  (read-only) => The shape of the selection region.
                               Not guaranteed to fit within "shape", although
                               the total number of points is less than
                               product(shape).
        fields (read-only) => fields included in the selection (for compound types)
                              if None, all fields are included
        nselect (read-only) => Number of selected points.  Always equal to
                               product(mshape).

        broadcast(target_shape) => Return an iterable which yields dataspaces
                                   for read, based on target_shape.

        The base class represents "unshaped" selections (1-D).
    """

    def __init__(self, shape, fields=None):
        """ Create a selection.   """

        shape = tuple(shape)
        self._shape = shape
        if fields is None:
            self._fields = None
        else:
            self._fields = set(fields)

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
    def fields(self):
        """ Fields of a compound type included in the selection """
        return self._fields

    @property
    def bbox(self):
        """ Bounding box of selection, as a tuple of (min, max) corner coordinates.

        For point-based selections, this is the smallest hyperslab that contains
        all selected points.  For hyperslab-based selections, this is the
        smallest hyperslab that contains the selection (which may be larger than
        the actual selection if stepped slices are used).
        """
        if self._select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
            slices = self.slices  # tuple of (slice|list|int) per dim
            mins, maxs = [], []
            for s in slices:
                if isinstance(s, list):
                    if not s:
                        return None, None
                    mins.append(min(s))
                    maxs.append(max(s) + 1)
                elif isinstance(s, slice):
                    if s.start == s.stop:
                        return None, None
                    mins.append(s.start)
                    maxs.append(s.stop)
                else:  # int
                    mins.append(s)
                    maxs.append(s + 1)
            return tuple(mins), tuple(maxs)
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


class SimpleSelection(Selection):

    """A selection composed of slices, integers, and/or coordinate lists.

    For pure slice/integer arguments the select_type is H5S_SEL_HYPERSLABS
    (or H5S_SEL_ALL when no arguments are supplied).  When any dimension is
    given as a list of coordinates or a boolean index array, the select_type
    is H5S_SEL_FANCY.  The start/count/step properties and broadcast() are
    only valid for hyperslab selections.
    """

    # --- Properties ---

    @property
    def mshape(self):
        """ Shape of current selection """
        return self._mshape

    @property
    def tgtshape(self):
        """ Shape of selection in rank of dataspace """
        if self._select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
            return list(self._mshape)
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

    @property
    def slices(self):
        """ Per-dimension slice/list/int components of the selection. """
        if self._select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
            return tuple(self._slices)
        rank = len(self._shape)
        return tuple(
            slice(self.start[d], self.start[d] + self.count[d], self.step[d])
            for d in range(rank)
        )

    # --- Initializer ---

    def __init__(self, shape, hyperslab=None, fields=None):
        Selection.__init__(self, shape, fields=fields)
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
            return self

        def _is_fancy_arg(arg):
            if isinstance(arg, (slice, type(Ellipsis))):
                return False
            if isinstance(arg, (list, tuple, np.ndarray)):
                return True
            try:
                int(arg)
                return False
            except (TypeError, ValueError):
                return True

        if any(_is_fancy_arg(a) for a in hyperslab):
            if len(hyperslab) != rank:
                raise TypeError("Number of coordinate sets does not match dataset rank")
            # Fancy path: process per-dimension slices, coordinate lists, and ints.
            select_type = H5S_SEL_HYPERSLABS  # upgraded to FANCY when a coord list is found
            slices = []
            mshape = []
            num_coordinates = None
            for idx in range(rank):
                length = self._shape[idx]
                arg = hyperslab[idx]
                if isinstance(arg, slice):
                    _, count, _ = _translate_slice(arg, length)
                    start = 0 if arg.start is None else arg.start
                    stop = length if arg.stop is None else arg.stop
                    step = 1 if arg.step is None else arg.step
                    slices.append(slice(start, stop, step))
                    mshape.append(count)
                    select_type = H5S_SEL_FANCY  # have both coordinates and slices
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
                elif isinstance(arg, list) or isinstance(arg, tuple) or hasattr(arg, 'dtype'):
                    slices.append(list(arg))
                    for x in arg:
                        if x < 0 or x >= length:
                            raise IndexError(f"Index ({arg}) out of range (0-{length - 1})")
                    if select_type == H5S_SEL_HYPERSLABS:
                        select_type = H5S_SEL_POINTS  # will set to FANCY if a slice is found
                    if num_coordinates is None:
                        num_coordinates = len(arg)
                    elif num_coordinates == len(arg):
                        # second coord list doesn't add to mshape
                        continue
                    else:
                        raise ValueError("coordinate num element missmatch")
                    mshape.append(len(arg))

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
        else:
            # Hyperslab path: slices and integer indices only.
            self._sel = _handle_simple(self._shape, hyperslab)
            self._mshape = tuple(x for x, y in zip(self._sel[1], self._sel[3]) if not y)
            self._select_type = H5S_SEL_HYPERSLABS

    # --- Methods ---

    def getSelectNpoints(self):
        """Return number of elements in current selection."""
        if self._select_type == H5S_SEL_NONE:
            return 0
        if self._select_type == H5S_SEL_ALL:
            npoints = 1
            for n in self._shape:
                npoints *= n
            return npoints
        if self._select_type == H5S_SEL_HYPERSLABS:
            npoints = 1
            for c in self.count:
                npoints *= c
            return npoints
        # H5S_SEL_FANCY — use _mshape which is set correctly for both Cartesian
        # (slice+list) and paired-coordinate (all-list) selections.
        npoints = 1
        for m in self._mshape:
            npoints *= m
        return npoints

    def getQueryParam(self):
        """ Get select param for use with HDF Rest API"""
        rank = len(self._shape)
        if rank == 0:
            return None
        query = ['[']
        if self._select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
            for dim, s in enumerate(self._slices):
                if isinstance(s, slice):
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
                    query.append(str(s))
                if dim + 1 < rank:
                    query.append(',')
        else:
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
                query.append(dim_sel)
        query.append(']')
        return "".join(query)

    def broadcast(self, target_shape):
        """ Return an iterator over target dataspaces for broadcasting.

        Only supported for hyperslab selections.
        """
        if self._select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
            raise TypeError("Broadcasting is not supported for complex selections")

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

    def __repr__(self):
        if self.fields:
            fields = ", fields: " + str(self.fields)
        else:
            fields = ""
        if self._select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
            return f"SimpleSelection(shape:{self._shape}, slices: {self._slices} {fields})"
        s = f"SimpleSelection(shape:{self._shape}, start: {self._sel[0]},"
        s += f" count: {self._sel[1]}, step: {self._sel[2]}"
        s += fields

        return s


# Backward-compatible alias
FancySelection = SimpleSelection

# Point selections are now represented as paired-coordinate FancySelections.
PointSelection = SimpleSelection
_empty_point_sel = _empty_paired_sel  # backward-compat alias


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
