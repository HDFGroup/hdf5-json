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

import itertools
import json
import struct

import numpy as np


# Selection types
H5S_SEL_NONE = 0
H5S_SEL_POINTS = 1
H5S_SEL_HYPERSLABS = 2
H5S_SEL_ALL = 3
H5S_SEL_FANCY = 4


# Boolean selection operations (values match the real HDF5 H5Sselect_t enum,
# i.e. h5py.h5s.SELECT_*)
H5S_SELECT_SET = 0
H5S_SELECT_OR = 1
H5S_SELECT_AND = 2
H5S_SELECT_XOR = 3
H5S_SELECT_NOTB = 4
H5S_SELECT_NOTA = 5
H5S_SELECT_APPEND = 6
H5S_SELECT_PREPEND = 7


# --- Binary (tobytes/frombytes) serialization format ---
#
# A compact, fixed-width binary layout - as opposed to to_dict()/JSON, which
# spells out coordinate lists as decimal text.  Point/fancy selections with
# large coordinate lists are stored as raw little-endian integer arrays (via
# numpy), avoiding per-value text formatting/parsing.
#
# All numeric payload values (shape extents, slice start/stop/step, scalar
# indices, and point-list coordinates) are bounded by the dataspace's
# extents, so a single unsigned integer width - chosen from the largest
# extent in the shape - is used for all of them: 16-bit when every extent is
# under 64K, 32-bit when under 4G, and 64-bit otherwise.
_SEL_MAGIC = b"HSEL"
_SEL_VERSION = 1
_SEL_CLASS_CODES = {"SimpleSelection": 0}
_SEL_DIM_SLICE = 0
_SEL_DIM_LIST = 1
_SEL_DIM_INT = 2

# width_code -> (byte width, struct format char, numpy dtype)
_SEL_WIDTH_INFO = {
    0: (2, "H", np.dtype("<u2")),
    1: (4, "I", np.dtype("<u4")),
    2: (8, "Q", np.dtype("<u8")),
}


def _select_width_code(shape):
    """ Smallest unsigned int width (see _SEL_WIDTH_INFO) that can hold every extent in shape. """
    max_extent = max(shape) if shape else 0
    if max_extent < (1 << 16):
        return 0
    elif max_extent < (1 << 32):
        return 1
    else:
        return 2


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

    if hasattr(obj, "shape"):
        obj_shape = obj.shape
    elif isinstance(obj, tuple):
        obj_shape = obj
    else:
        raise TypeError("Object must be a dataset or a shape tuple")

    if isinstance(args, dict):
        args = _handle_dict_selection(obj_shape, args)
    elif not isinstance(args, tuple):
        args = (args,)

    # "Special" indexing objects - checked before the scalar-shape early
    # return below, since a RegionReference (or an existing Selection) can
    # legitimately be created against a scalar dataspace too (e.g. a
    # select_none() vs select_all() region reference on a scalar dataset).
    if len(args) == 1:

        arg = args[0]

        if isinstance(arg, str):
            # convert seleection_str to tuple of slices/coordinates
            if arg == "[...]":
                args = (...,)
            else:
                args = _getSelectionList(obj_shape, arg)

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

        elif arg.__class__.__name__ == "RegionReference":
            if arg.id is None:
                raise ValueError("Cannot select using a null region reference")
            obj_id = getattr(getattr(obj, "id", None), "uuid", None)
            if obj_id is not None:
                from .objid import getHashTagForId
                if getHashTagForId(arg.id) != getHashTagForId(obj_id):
                    raise TypeError("Region reference must point to this dataset")
            if arg.selection_bytes is None:
                # no selection was bound - the whole dataset is referenced
                return SimpleSelection(obj_shape, fields=fields)
            ref_sel = Selection.frombytes(arg.selection_bytes)
            if ref_sel.shape != obj_shape:
                # A region reference that round-tripped through the JSON
                # attribute representation (RegionReference.from_json() ->
                # from_region_json()) only recovers the *minimal* bounding
                # shape of its selection, not the true dataset shape (see
                # from_region_json()'s docstring) - a region reference that
                # round-tripped through raw bytes (a dataset element) always
                # has the exact original shape. The per-dimension slices/
                # coordinates are still valid absolute indices either way,
                # so rebuild against the real shape instead of rejecting a
                # reference just because it came from an attribute.
                if len(ref_sel.shape) != len(obj_shape) or any(
                    r > o for r, o in zip(ref_sel.shape, obj_shape)
                ):
                    raise TypeError("Region reference shape does not match dataset shape")
                ref_sel = SimpleSelection(obj_shape, ref_sel.slices, fields=ref_sel.fields)
            if fields is not None:
                ref_sel._fields = set(fields)
            return ref_sel

    sel = SimpleSelection(obj_shape, args, fields=fields)
    return sel


def _check_bool_args(s1, s2):
    """ verify argument for boolean operations """
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


def _getSelectElements(sel_str):
    """helper method - return array of queries for each
    dimension"""
    if not isinstance(sel_str, str):
        raise TypeError("expected string arg")
    if len(sel_str) < 3:
        raise ValueError("selection string too short")
    if sel_str[0] != '[' or sel_str[-1] != ']':
        raise ValueError("unexpected selection string format")
    sel_str = sel_str[1:-1]  # strip brackets

    query_array = []
    dim_query = []
    coord_list = False
    for ch in sel_str:
        if ch.isspace():
            # ignore
            pass
        elif ch == ",":
            if coord_list:
                dim_query.append(ch)
            else:
                if len(dim_query) == 0:
                    # empty dimension
                    raise ValueError("invalid query")
                query_array.append("".join(dim_query))
                dim_query = []  # reset
        elif ch == "[":
            if coord_list:
                # can't have nested coordinates
                raise ValueError("invalid query")
            coord_list = True
            dim_query.append(ch)
        elif ch == "]":
            if not coord_list:
                # close bracket with no open
                raise ValueError("invalid query")
            dim_query.append(ch)
            coord_list = False
        elif ch == ":":
            if coord_list:
                # range not allowed in coord list
                raise ValueError("invalid query")
            dim_query.append(ch)
        else:
            dim_query.append(ch)
    if not dim_query:
        # empty dimension
        raise ValueError("invalid query")
    query_array.append("".join(dim_query))

    return query_array


def _getSelectionList(shape, sel_str):
    """Return tuple of slices and/or coordinate list for the given selection"""
    select_list = []

    if sel_str is None or len(sel_str) == 0:
        """Return set of slices covering data space"""
        slices = []
        for extent in shape:
            s = slice(0, extent, 1)
            slices.append(s)
        return tuple(slices)

    # convert selection to list by dimension
    elements = _getSelectElements(sel_str)
    rank = len(elements)
    if len(shape) != rank:
        raise ValueError("invalid rank for selection")
    for dim in range(rank):
        extent = shape[dim]
        element = elements[dim]
        is_list = isinstance(element, list)
        is_str = isinstance(element, str)
        if is_list or (is_str and element.startswith("[")):
            # list of coordinates
            if is_str:
                fields = element[1:-1].split(",")
            else:
                fields = element
            coords = []
            for field in fields:
                if isinstance(field, str) and not field:
                    continue
                try:
                    coord = int(field)
                except ValueError:
                    raise ValueError(f"Invalid coordinate for dim {dim}")
                if coord < 0 or coord >= extent:
                    msg = f"out of range coordinate for dim {dim}, {coord} "
                    msg += f"not in range: 0-{extent - 1}"
                    raise ValueError(msg)
                coords.append(coord)
            select_list.append(coords)
        elif element == ":":
            s = slice(0, extent, 1)
            select_list.append(s)
        elif is_str and element.find(":") >= 0:
            fields = element.split(":")
            if len(fields) not in (2, 3):
                raise ValueError(f"Invalid selection format for dim {dim}")
            if len(fields[0]) == 0:
                start = 0
            else:
                try:
                    start = int(fields[0])
                except ValueError:
                    raise ValueError(f"Invalid selection - start value for dim {dim}")
                if start < 0 or start >= extent:
                    msg = f"Invalid selection - start value out of range for dim {dim}"
                    raise ValueError(msg)
            if len(fields[1]) == 0:
                stop = extent
            else:
                try:
                    stop = int(fields[1])
                except ValueError:
                    raise ValueError(f"Invalid selection - stop value for dim {dim}")
                if stop < 0 or stop > extent or stop <= start:
                    msg = f"Invalid selection - stop value out of range for dim {dim}"
                    raise ValueError(msg)
            if len(fields) == 3:
                # get step value
                if len(fields[2]) == 0:
                    step = 1
                else:
                    try:
                        step = int(fields[2])
                    except ValueError:
                        msg = f"Invalid selection - step value for dim {dim}"
                        raise ValueError(msg)
                    if step <= 0:
                        msg = f"Invalid selection - step value out of range for dim {dim}"
                        raise ValueError(msg)
            else:
                step = 1
            s = slice(start, stop, step)
            select_list.append(s)
        else:
            # expect single coordinate value
            try:
                index = int(element)
            except ValueError:
                raise ValueError(f"Invalid selection - index value for dim {dim}")
            if index < 0 or index >= extent:
                msg = f"Invalid selection - index value out of range for dim {dim}"
                raise ValueError(msg)
            s = slice(index, index + 1, 1)
            select_list.append(s)
    # end dimension loop
    return tuple(select_list)


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
    """Yield each point in a paired-coordinate fancy selection as a tuple of ints."""
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
    """Return an empty paired-coordinate fancy selection for the given shape."""
    rank = len(shape)
    return SimpleSelection(shape, tuple([] for _ in range(rank)))


def from_query_result(shape, indices):
    """Create a PointSelection from an arrayQuery result.

    shape: full dataset shape tuple
    indices: ndarray of shape (N, rank), as returned by arrayQuery
    """
    rank = len(shape)
    if len(indices) == 0:
        return _empty_paired_sel(shape)
    if rank == 1:
        return select(shape, indices[:, 0].astype(int).tolist())
    coords = tuple(indices[:, d].astype(int).tolist() for d in range(rank))
    return select(shape, coords)


def _slice_to_ap(s):
    """Return (start, step, count) for a slice with concrete (non-None)
    int start/stop/step, as produced by Selection.slices."""
    if s.stop <= s.start:
        return s.start, s.step, 0
    return s.start, s.step, 1 + (s.stop - s.start - 1) // s.step


def _ap_to_slice(start, step, count):
    """Inverse of _slice_to_ap(): build a slice from (start, step, count)."""
    if count <= 0:
        return slice(start, start, 1)
    return slice(start, start + (count - 1) * step + 1, step)


def _extended_gcd(a, b):
    """Return (g, x, y) such that a*x + b*y == g == gcd(a, b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    return old_r, old_s, (old_r - a * old_s) // b if b else 0


def _intersect_stepped_range(a1, d1, n1, a2, d2, n2):
    """Return (start, step, count) describing the intersection of the two
    arithmetic progressions {a1 + i*d1 : 0 <= i < n1} and
    {a2 + j*d2 : 0 <= j < n2} (each representing one dimension of a
    hyperslab selection), or None if they don't intersect at all.

    The intersection of two arithmetic progressions is itself an arithmetic
    progression (or empty) - found via the Chinese Remainder Theorem: solve
    x = a1 (mod d1), x = a2 (mod d2) for the common step (lcm(d1, d2)), then
    clip to the overlap of the two progressions' own ranges.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    end1 = a1 + (n1 - 1) * d1
    end2 = a2 + (n2 - 1) * d2
    lo = max(a1, a2)
    hi = min(end1, end2)
    if lo > hi:
        return None

    g, p, _ = _extended_gcd(d1, d2)
    diff = a2 - a1
    if diff % g != 0:
        return None  # progressions never share a common value

    lcm = d1 // g * d2
    remainder = (a1 + d1 * ((diff // g) * p)) % lcm

    # smallest x >= lo with x % lcm == remainder
    start = lo + ((remainder - lo) % lcm)
    if start > hi:
        return None
    count = (hi - start) // lcm + 1
    return start, lcm, count


def _ap_is_subset(a1, d1, n1, a2, d2, n2):
    """Return True if every value of AP1 = {a1 + i*d1 : 0 <= i < n1} is also
    a value of AP2 = {a2 + j*d2 : 0 <= j < n2}."""
    if n1 <= 0:
        return True  # vacuously true - nothing to contain
    if n1 == 1:
        # a single point - its own step doesn't matter
        return a2 <= a1 <= a2 + (n2 - 1) * d2 and (a1 - a2) % d2 == 0
    if d1 % d2 != 0 or (a1 - a2) % d2 != 0:
        return False
    end1 = a1 + (n1 - 1) * d1
    end2 = a2 + (n2 - 1) * d2
    return a1 >= a2 and end1 <= end2


def _intersect_paired_fancy(s1, s2):
    """Return the intersection of two paired-coordinate fancy selections."""
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
    """Return the intersection of a fancy selection with a hyperslab selection.

    For Cartesian-product selections (at most one list dimension) each
    dimension is clipped independently.  For paired-coordinate selections
    (multiple list dimensions) the coordinate pairs are filtered as a unit so
    the two lists always retain the same length.  Returns an empty
    paired fancy selection when the intersection is empty.
    """
    rank = len(fancy_sel.shape)
    h_start = hyper_sel.start
    h_count = hyper_sel.count
    h_step = hyper_sel.step
    slices = fancy_sel.slices  # tuple after the property fix

    list_dims = [d for d in range(rank) if isinstance(slices[d], list)]

    if len(list_dims) > 1:
        # Paired-coordinate selection: check slice dims first, then filter pairs.
        slice_inter = {}
        for dim in range(rank):
            s = slices[dim]
            hs, hc, hst = h_start[dim], h_count[dim], h_step[dim]
            if isinstance(s, slice):
                s_start, s_step, s_count = _slice_to_ap(s)
                inter = _intersect_stepped_range(s_start, s_step, s_count, hs, hst, hc)
                if inter is None:
                    return _empty_paired_sel(fancy_sel.shape)
                slice_inter[dim] = inter
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
                new_slices.append(_ap_to_slice(*slice_inter[dim]))
            else:  # int: already validated above, keep as-is
                new_slices.append(s)
        return SimpleSelection(fancy_sel.shape, new_slices)

    # Cartesian-product path: clip each dimension independently.
    new_slices = []
    for dim in range(rank):
        s = slices[dim]
        hs = h_start[dim]
        hc = h_count[dim]
        hst = h_step[dim]

        if isinstance(s, slice):
            s_start, s_step, s_count = _slice_to_ap(s)
            inter = _intersect_stepped_range(s_start, s_step, s_count, hs, hst, hc)
            if inter is None:
                return _empty_paired_sel(fancy_sel.shape)
            new_slices.append(_ap_to_slice(*inter))
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
            raise TypeError(f"Unexpected selection slice type: {type(s)}")

    return SimpleSelection(fancy_sel.shape, new_slices)


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
            inter = _intersect_stepped_range(
                s1.start[dim], s1.step[dim], s1.count[dim],
                s2.start[dim], s2.step[dim], s2.count[dim],
            )
            if inter is None:
                slices.append(slice(0, 0, 1))
            else:
                slices.append(_ap_to_slice(*inter))
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


def _dim_ap(dim):
    """Normalise a slice to (start, step, count) - defaulting an unset
    start/step to 0/1, as _dim_contained's caller (_fancy_contained) may
    pass a slice built directly from raw args rather than via .slices."""
    start = dim.start if dim.start is not None else 0
    step = dim.step if dim.step is not None else 1
    stop = dim.stop
    count = 0 if stop <= start else 1 + (stop - start - 1) // step
    return start, step, count


def _dim_contained(s1_dim, s2_dim):
    """Return True if every value represented by s1_dim is also in s2_dim.

    Each argument is a per-dimension component: a slice, list of ints, or int.
    """
    # Normalise s1 to either an arithmetic progression or an explicit set.
    if isinstance(s1_dim, int):
        s1_ap, s1_set = (s1_dim, 1, 1), None
    elif isinstance(s1_dim, list):
        s1_ap, s1_set = None, set(s1_dim)
    elif isinstance(s1_dim, slice):
        s1_ap, s1_set = _dim_ap(s1_dim), None
    else:
        return False

    if isinstance(s2_dim, slice):
        s2_ap = _dim_ap(s2_dim)
        if s1_ap is not None:
            return _ap_is_subset(*s1_ap, *s2_ap)
        else:
            return all(_ap_is_subset(x, 1, 1, *s2_ap) for x in s1_set)
    elif isinstance(s2_dim, list):
        s2_set = set(s2_dim)
        if s1_ap is not None:
            start, step, count = s1_ap
            return all((start + i * step) in s2_set for i in range(count))
        else:
            return s1_set <= s2_set
    elif isinstance(s2_dim, int):
        if s1_ap is not None:
            start, _, count = s1_ap
            return count <= 1 and (count == 0 or start == s2_dim)
        else:
            return s1_set == {s2_dim}
    else:
        return False


def _fancy_contained(s1, s2):
    """Return True if every element selected by s1 is also selected by s2.

    At least one of s1/s2 must be a fancy selection; the other may be a
    SimpleSelection (hyperslab or select-all).

    fancy selections with multiple list dimensions represent paired (non-grid)
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
        else:  # H5S_SEL_FANCY or H5S_SEL_POINTS
            return list(sel.slices)

    s1_dims = get_dims(s1)
    s2_dims = get_dims(s2)

    # Paired-coordinate fancy selections (multiple list dims) are not a
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
    fancy_types = (H5S_SEL_FANCY, H5S_SEL_POINTS)
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
        if not _ap_is_subset(s1.start[dim], s1.step[dim], s1.count[dim],
                             s2.start[dim], s2.step[dim], s2.count[dim]):
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
            raise TypeError(f"translate with fancy selection s1 does not support s2 type: {s2.select_type}")
        if s1.shape != s2.shape:
            raise ValueError("selections have incompatible shapes")

        rank = len(s1.shape)

        # Compute the intersection in absolute coordinates.
        if s2.select_type in hyperslab_types:
            sel_inter = intersect(s1, s2)
            if sel_inter.nselect == 0:
                raise ValueError("translate - selections not overlapping")
            inter_slices = sel_inter.slices
        else:  # s2 is also a fancy selection
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
        return SimpleSelection(s1.shape, new_slices)

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
        return SimpleSelection(s1.shape, new_slices)

    _check_bool_args(s1, s2)
    sel_inter = intersect(s1, s2)
    if sel_inter.nselect == 0:
        raise ValueError("translate - selections not overlapping")

    rank = len(s1.shape)
    args = []
    if s2.select_type == H5S_SEL_HYPERSLABS:
        for dim in range(rank):
            # s2 (contained in s1) is expressed in s1's own *dense* selected-
            # index frame, not raw absolute offset - e.g. if s1 is
            # start=1,step=3 (selecting 1,4,7,...), s1's local index 0 is
            # absolute position 1, local index 1 is absolute position 4, and
            # so on - so both the offset and s2's step need dividing by
            # s1's step to land on the right dense index.
            s1_step = s1.step[dim]
            offset = s2.start[dim] - s1.start[dim]
            args.append(_ap_to_slice(offset // s1_step, s2.step[dim] // s1_step, s2.count[dim]))
    else:
        raise TypeError("translate - unsupported selection type for s2")
    return select(s1.shape, tuple(args))


def _handle_dict_selection(shape, arg):
    """ Handle a dictionary-based selection, where the keys are dimension indices and
    the values are slices or lists of coordinates. Returns a tuple of slices/lists for
    each dimension, filling in full slices for unspecified dimensions.
    """
    rank = len(shape)
    slices = []
    if "start" not in arg:
        start = (0,) * rank
    else:
        start = arg["start"]
    if "stop" not in arg:
        stop = shape
    else:
        stop = arg["stop"]
    if "step" not in arg:
        step = (1,) * rank
    else:
        step = arg["step"]

    if isinstance(start, int):
        start = (start,) * rank  # broadcast to all dimensions
    elif isinstance(start, (list, tuple)):
        if len(start) != rank:
            raise ValueError("Start list length does not match dataset rank")
    else:
        raise TypeError("Start value must be an int or a list/tuple of ints")

    if isinstance(stop, int):
        stop = (stop,) * rank  # broadcast to all dimensions
    elif isinstance(stop, (list, tuple)):
        if len(stop) != rank:
            raise ValueError("Stop list length does not match dataset rank")
    else:
        raise TypeError("Stop value must be an int or a list/tuple of ints")

    if isinstance(step, int):
        step = (step,) * rank  # broadcast to all dimensions
    elif isinstance(step, (list, tuple)):
        if len(step) != rank:
            raise ValueError("Step list length does not match dataset rank")
    else:
        raise TypeError("Step value must be an int or a list/tuple of ints")

    for idx in range(rank):
        s = slice(start[idx], stop[idx], step[idx])
        slices.append(s)

    return slices


def from_dict(d):
    """ Reconstruct a Selection instance from a dict produced by Selection.to_dict(). """
    shape = tuple(d["shape"])
    fields = d.get("fields")
    cls_name = d.get("class", "SimpleSelection")

    if cls_name != "SimpleSelection":
        raise ValueError(f"Unsupported selection class: {cls_name}")

    if d.get("select_type") == H5S_SEL_ALL:
        return SimpleSelection(shape, None, fields=fields)

    args = []
    for item in d["slices"]:
        item_type = item["type"]
        if item_type == "slice":
            args.append(slice(item["start"], item["stop"], item["step"]))
        elif item_type == "list":
            args.append(list(item["values"]))
        elif item_type == "int":
            args.append(item["value"])
        else:
            raise ValueError(f"Unsupported slice element type: {item_type}")
    return SimpleSelection(shape, tuple(args), fields=fields)


def from_region_json(d):
    """ Reconstruct a Selection from the {"select_type": ..., "selection": [...]}
    representation used for HDF5 region references in the h5json format (the
    mirror of SimpleSelection.to_region_json()) - see
    data/json/regionref_dset.json for an example.

    This representation doesn't carry the referenced dataset's shape, so a
    minimal shape that just contains the given selection is used - the
    resulting sel.shape will not necessarily equal the true referenced
    dataset's shape.

    A single hyperslab block reconstructs as H5S_SEL_HYPERSLABS.  Multiple
    disjoint blocks (as a real HDF5 region reference can have, e.g. for a
    blocked/strided hyperslab - this project's Selection model has no
    equivalent for that) are expanded into an equivalent paired-coordinate
    point selection covering the exact same cells; this can be memory
    intensive for selections with very large blocks.
    """
    select_type = d.get("select_type")
    selection = d.get("selection")
    if not selection:
        raise ValueError("Empty region reference selection")

    if select_type == "H5S_SEL_POINTS":
        rank = len(selection[0])
        shape = tuple(max(int(pt[dim]) for pt in selection) + 1 for dim in range(rank))
        coords = tuple([int(pt[dim]) for pt in selection] for dim in range(rank))
        return select(shape, coords)

    if select_type == "H5S_SEL_HYPERSLABS":
        rank = len(selection[0][0])
        if len(selection) == 1:
            start, end = selection[0]
            shape = tuple(int(e) + 1 for e in end)
            args = tuple(slice(int(s), int(e) + 1, 1) for s, e in zip(start, end))
            return select(shape, args)

        # multiple disjoint blocks - no single-hyperslab equivalent in this
        # model, so expand to the exact set of covered points instead
        points = []
        for start, end in selection:
            ranges = [range(int(s), int(e) + 1) for s, e in zip(start, end)]
            points.extend(itertools.product(*ranges))
        shape = tuple(max(pt[dim] for pt in points) + 1 for dim in range(rank))
        coords = tuple([pt[dim] for pt in points] for dim in range(rank))
        return select(shape, coords)

    raise NotImplementedError(f"Region reference JSON import not supported for select_type {select_type}")


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
        if np.prod(target_shape) != self.nselect:
            raise TypeError("Broadcasting is not supported for point-wise selections")
        yield self._id

    def __getitem__(self, args):
        raise NotImplementedError("This class does not support indexing")

    def __eq__(self, other):
        if not isinstance(other, Selection):
            return NotImplemented
        return all((
            type(self) is type(other),
            self.shape == other.shape,
            self.select_type == other.select_type,
            self.fields == other.fields,
            self.mshape == other.mshape,
        ))

    def __repr__(self):
        return f"Selection(shape:{self._shape})"

    def to_dict(self):
        """ Return a JSON-serializable dict representation of this selection. """
        d = {
            "class": type(self).__name__,
            "shape": list(self._shape),
            "select_type": self._select_type,
        }
        if self._fields is not None:
            d["fields"] = sorted(self._fields)
        return d

    def _pack_body(self, width_code):
        """ Subclass hook: return the binary payload specific to this selection type. """
        return b""

    def tobytes(self):
        """ Serialize this selection to a compact binary bytearray.

        Unlike to_dict()/JSON, coordinate lists (fancy/point selections) are
        stored as raw integer arrays rather than decimal text, so this stays
        cheap for selections with large numbers of points.  The integer
        width used for shape/index values (16/32/64-bit) is chosen from the
        selection's own extents - see _select_width_code().
        """
        width_code = _select_width_code(self._shape)
        _, fmt, _ = _SEL_WIDTH_INFO[width_code]
        buf = bytearray()
        buf += _SEL_MAGIC
        buf += struct.pack("<BBBBH", _SEL_VERSION, _SEL_CLASS_CODES[type(self).__name__],
                           self._select_type, width_code, len(self._shape))
        if self._shape:
            buf += struct.pack(f"<{len(self._shape)}{fmt}", *self._shape)
        if self._fields is None:
            buf += struct.pack("<B", 0)
        else:
            fields_sorted = sorted(self._fields)
            buf += struct.pack("<BH", 1, len(fields_sorted))
            for f in fields_sorted:
                fb = f.encode("utf-8")
                buf += struct.pack("<H", len(fb))
                buf += fb
        buf += self._pack_body(width_code)
        return buf

    @classmethod
    def frombytes(cls, data):
        """ Reconstruct a Selection instance from a bytearray produced by tobytes(). """
        data = bytes(data)
        if data[:4] != _SEL_MAGIC:
            raise ValueError("Invalid selection byte stream")
        version, class_code, select_type, width_code, rank = struct.unpack_from("<BBBBH", data, 4)
        offset = 4 + struct.calcsize("<BBBBH")
        if version != _SEL_VERSION:
            raise ValueError(f"Unsupported selection serialization version: {version}")
        if width_code not in _SEL_WIDTH_INFO:
            raise ValueError(f"Unsupported selection integer width code: {width_code}")
        _, fmt, _ = _SEL_WIDTH_INFO[width_code]

        if rank:
            shape = struct.unpack_from(f"<{rank}{fmt}", data, offset)
            offset += _SEL_WIDTH_INFO[width_code][0] * rank
        else:
            shape = ()

        fields_flag = data[offset]
        offset += 1
        fields = None
        if fields_flag:
            count = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            fields = []
            for _ in range(count):
                flen = struct.unpack_from("<H", data, offset)[0]
                offset += 2
                fields.append(data[offset:offset + flen].decode("utf-8"))
                offset += flen

        class_name = {v: k for k, v in _SEL_CLASS_CODES.items()}.get(class_code)
        if class_name is None:
            raise ValueError(f"Unsupported selection class code: {class_code}")
        target_cls = globals()[class_name]
        return target_cls._unpack_body(shape, select_type, fields, width_code, data, offset)


class SimpleSelection(Selection):

    """A selection composed of slices, integers, and/or coordinate lists.

    For pure slice/integer arguments the select_type is H5S_SEL_HYPERSLABS
    (or H5S_SEL_ALL when no arguments are supplied).  When any dimension is
    given as a list of coordinates or a boolean index array, the select_type
    is H5S_SEL_FANCY.  The start/count/step properties and broadcast() are
    only valid for hyperslab selections.

    A scalar dataset (shape == ()) is also represented by this class: it
    has exactly one point, so any valid construction (None, (), (Ellipsis,))
    selects it and always yields select_type H5S_SEL_ALL.
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
    def scalar(self):
        """ Per-dimension flags: True where that dimension was indexed by a
        bare integer (and so is excluded from mshape - see numpy's basic
        indexing rules). Only meaningful for a plain hyperslab selection
        (select_type H5S_SEL_ALL or H5S_SEL_HYPERSLABS) - a fancy/points
        selection's mshape already has integer-indexed dimensions removed,
        with nothing left for a caller to additionally drop. """
        return self._sel[3]

    @property
    def slices(self):
        """ Per-dimension slice/list/int components of the selection. """
        if self._select_type in (H5S_SEL_FANCY, H5S_SEL_POINTS):
            return tuple(self._slices)
        rank = len(self._shape)
        return tuple(
            slice(self.start[d], self.start[d] + self.count[d] * self.step[d], self.step[d])
            for d in range(rank)
        )

    # --- Initializer ---

    def __init__(self, shape, hyperslab=None, fields=None):
        Selection.__init__(self, shape, fields=fields)
        rank = len(self._shape)

        if hyperslab is None:
            self._sel = ((0,) * rank, self._shape, (1,) * rank, (False,) * rank)
            self._mshape = self._shape
            self._select_type = H5S_SEL_ALL
            return

        if any(a is Ellipsis for a in hyperslab):
            # _handle_simple() (the non-fancy path below) expands Ellipsis
            # internally, but the fancy path's own rank check runs before
            # that - so a fancy arg combined with Ellipsis (e.g. [0], ...)
            # needs it expanded here too, or the raw (unexpanded) length
            # mismatches the dataset rank.
            hyperslab = tuple(_expand_ellipsis(hyperslab, rank))

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
            # A scalar (rank-0) dataspace has exactly one point - any valid
            # selection on it (None, (), (Ellipsis,), ...) selects that point,
            # so canonicalize to ALL regardless of which form was given.
            self._select_type = H5S_SEL_ALL if rank == 0 else H5S_SEL_HYPERSLABS

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

    @property
    def query_string(self):
        """ The value of the 'select' query parameter for this selection, for use with the HDF REST API """
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
            if np.prod(target_shape) != 1:
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

    def __eq__(self, other):
        if not isinstance(other, SimpleSelection):
            return NotImplemented
        return all((
            self.shape == other.shape,
            self.select_type == other.select_type,
            self.fields == other.fields,
            self.slices == other.slices,
        ))

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

    def to_dict(self):
        d = Selection.to_dict(self)
        if self._select_type != H5S_SEL_ALL:
            slices_out = []
            for s in self.slices:
                if isinstance(s, slice):
                    s = {"type": "slice", "start": int(s.start), "stop": int(s.stop), "step": int(s.step)}
                elif isinstance(s, list):
                    s = {"type": "list", "values": [int(x) for x in s]}
                else:
                    s = {"type": "int", "value": int(s)}
                slices_out.append(s)
            d["slices"] = slices_out
        return d

    def to_region_json(self):
        """ Convert this selection to the {"select_type": ..., "selection": [...]}
        representation used for HDF5 region references in the h5json format
        (see data/json/regionref_dset.json for an example).

        Only paired-coordinate point selections (H5S_SEL_POINTS) and
        unit-step hyperslab selections (H5S_SEL_HYPERSLABS/H5S_SEL_ALL) are
        supported, since those are the only forms this project's Selection
        model can represent as points or as a single contiguous block.
        Mixed slice/coordinate selections (H5S_SEL_FANCY) and stepped
        hyperslab selections have no equivalent in this format.
        """
        if self._select_type == H5S_SEL_POINTS:
            points = [list(pt) for pt in _iter_points(self)]
            return {"select_type": "H5S_SEL_POINTS", "selection": points}

        if self._select_type in (H5S_SEL_HYPERSLABS, H5S_SEL_ALL):
            if any(step != 1 for step in self.step):
                raise NotImplementedError(
                    "Region reference JSON export does not support stepped hyperslab selections"
                )
            start = list(self.start)
            end = [s + c - 1 for s, c in zip(self.start, self.count)]
            return {"select_type": "H5S_SEL_HYPERSLABS", "selection": [[start, end]]}

        raise NotImplementedError(
            f"Region reference JSON export not supported for select_type {self._select_type}"
        )

    def _pack_body(self, width_code):
        if self._select_type == H5S_SEL_ALL:
            return b""
        width, fmt, np_dtype = _SEL_WIDTH_INFO[width_code]
        buf = bytearray()
        for s in self.slices:
            if isinstance(s, slice):
                buf += struct.pack(f"<B{fmt}{fmt}{fmt}", _SEL_DIM_SLICE, s.start, s.stop, s.step)
            elif isinstance(s, list):
                arr = np.asarray(s, dtype=np_dtype)
                buf += struct.pack("<BI", _SEL_DIM_LIST, arr.size)
                buf += arr.tobytes()
            else:
                buf += struct.pack(f"<B{fmt}", _SEL_DIM_INT, int(s))
        return buf

    @classmethod
    def _unpack_body(cls, shape, select_type, fields, width_code, data, offset):
        if select_type == H5S_SEL_ALL:
            return cls(shape, None, fields=fields)
        width, fmt, np_dtype = _SEL_WIDTH_INFO[width_code]
        args = []
        for _ in range(len(shape)):
            dim_type = data[offset]
            offset += 1
            if dim_type == _SEL_DIM_SLICE:
                start, stop, step = struct.unpack_from(f"<{fmt}{fmt}{fmt}", data, offset)
                offset += 3 * width
                args.append(slice(start, stop, step))
            elif dim_type == _SEL_DIM_LIST:
                n = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                arr = np.frombuffer(data, dtype=np_dtype, count=n, offset=offset)
                offset += width * n
                args.append(arr.tolist())
            elif dim_type == _SEL_DIM_INT:
                val = struct.unpack_from(f"<{fmt}", data, offset)[0]
                offset += width
                args.append(int(val))
            else:
                raise ValueError(f"Unsupported dim type: {dim_type}")
        return cls(shape, tuple(args), fields=fields)


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

    from h5py import h5s

    sel_class = sid.get_simple_extent_type()    # Dataspace class
    sel_type = sid.get_select_type()            # Flavor of selection in use

    if sel_class == h5s.NULL:
        # NULL dataspaces don't support selections
        return None

    elif sel_class == h5s.SCALAR:
        # NumPy has no way of expressing empty 0-rank selections, so we use None
        if sel_type == H5S_SEL_NONE:
            return None
        if sel_type == H5S_SEL_ALL:
            return tuple()

    elif sel_class != h5s.SIMPLE:
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

    if np.prod(shape) != N:
        # This means multiple hyperslab selections are in effect,
        # so we fall back to a 1D shape
        return (N,)

    return shape
