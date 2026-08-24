# selections

This module models HDF5 dataspace selections — hyperslabs (start/count/step), point selections (paired per-dimension coordinate lists), and "fancy" selections that mix ordinary slices with coordinate lists in different dimensions. A single concrete class, `SimpleSelection`, represents all of these variants, dispatching behavior on an internal `select_type` (`H5S_SEL_ALL`, `H5S_SEL_HYPERSLABS`, `H5S_SEL_POINTS`, or `H5S_SEL_FANCY`). Both storage backends (h5py and JSON) use this module to build selections from user arguments, to compute intersections/containment/translation between selections, and to serialize selections to/from a compact binary format (`tobytes`/`frombytes`) or to/from JSON (`to_dict`/`from_dict`, and `to_region_json`/`from_region_json` for the HDF5-region-reference JSON representation).

## Selection

Base class for HDF5 dataspace selections. It documents the "selection protocol" (a `shape`, `mshape`, `fields`, `nselect`, `broadcast()`, etc.) that subclasses are expected to implement, and itself implements the degenerate case of an unshaped, fully-selected (`H5S_SEL_ALL`) dataspace of 1 or more dimensions. It also implements the shared JSON (`to_dict`) and binary (`tobytes`/`frombytes`) serialization scaffolding, with subclasses supplying only the selection-specific payload via `_pack_body`/`_unpack_body`.

### Selection.select_type

Read-only property returning the internal selection-type code: one of `H5S_SEL_NONE`, `H5S_SEL_POINTS`, `H5S_SEL_HYPERSLABS`, `H5S_SEL_ALL`, or `H5S_SEL_FANCY`.

### Selection.shape

Read-only property returning the shape tuple of the full dataspace the selection was created against (not the shape of the selected region).

### Selection.fields

Read-only property returning the set of compound-type field names included in the selection, or `None` if all fields are included.

### Selection.bbox

Read-only property returning a `(min, max)` tuple of corner coordinates for the smallest hyperslab bounding box containing the selection. For point/fancy selections it is derived from the per-dimension `slices`, returning `(None, None)` if any list-valued dimension is empty or any slice dimension is empty (`start == stop`). For hyperslab/all selections it is computed from `start`/`count`/`step`, and may be larger than the actual selection when a stepped slice is used. Raises `TypeError` for any other selection type.

### Selection.nselect

Read-only property giving the number of currently selected elements; implemented as `self.getSelectNpoints()`.

### Selection.mshape

Read-only property giving the shape of the selection itself. On the base class this is always `(nselect,)` (one-dimensional).

### Selection.tgtshape

Read-only property giving the shape of the selection expressed in the rank of the dataspace. On the base class this is the same as `mshape`.

### Selection.getSelectNpoints()

Returns the number of selected points. On the base class this only supports `H5S_SEL_NONE` (returns 0) and `H5S_SEL_ALL` (returns the product of the shape's extents); any other `select_type` raises `IOError`.

### Selection.broadcast(target_shape)

Generator that yields dataspaces for reading, based on `target_shape`. On the base class it only supports the case where `target_shape`'s product equals `nselect` (raising `TypeError` otherwise), yielding `self._id` once.

### Selection.to_dict()

Returns a JSON-serializable `dict` with the selection's class name, `shape` (as a list), and `select_type`; `fields` is included (as a sorted list) only when field selection is in effect. Subclasses extend this with selection-specific keys (see `SimpleSelection.to_dict`).

### Selection.tobytes()

Serializes the selection to a compact `bytearray`, used as an alternative to JSON when coordinate lists may be large (avoids per-value decimal text formatting). The layout is: 4-byte magic `b"HSEL"`; then a `<BBBBH` struct of version (`_SEL_VERSION`, currently `1`), class code (from `_SEL_CLASS_CODES`, currently only `SimpleSelection` -> `0`), `select_type`, an integer-width code, and the dataspace rank; then the shape extents packed at that width; then a fields block (a `0` byte if `fields is None`, otherwise a `1` byte followed by a count and each field name's UTF-8 length-prefixed bytes); then the subclass-specific body from `_pack_body`. The width code (0/1/2, meaning 16/32/64-bit unsigned ints) is chosen by `_select_width_code` based on the largest extent in the shape, so shape values, slice start/stop/step, scalar indices, and point coordinates all use the smallest width that can hold every shape extent.

### Selection.frombytes(data)

Class method that reconstructs a `Selection` instance (in practice, a `SimpleSelection`) from bytes produced by `tobytes()`. Validates the magic bytes and version, decodes the shape/select_type/width/fields header, resolves the concrete class via `_SEL_CLASS_CODES`, and delegates the type-specific payload decoding to that class's `_unpack_body`. Raises `ValueError` for a bad magic number, unsupported version, unsupported width code, or unsupported class code.

## SimpleSelection

A selection composed of slices, integers, and/or coordinate lists (`Selection`'s only concrete subclass). When constructed from pure slice/integer arguments, `select_type` is `H5S_SEL_HYPERSLABS` (or `H5S_SEL_ALL` if no arguments were given). When any dimension is given as a coordinate list or boolean index array, `select_type` becomes `H5S_SEL_POINTS` (all list dimensions, paired coordinates) or `H5S_SEL_FANCY` (a mix of slice/int dimensions and list dimensions, i.e. a Cartesian-product-style selection). `start`/`count`/`step` and `broadcast()` are only meaningful for hyperslab/all selections. A scalar dataset (`shape == ()`) is always canonicalized to `H5S_SEL_ALL` regardless of which construction form (`None`, `()`, `(Ellipsis,)`) was used.

### SimpleSelection.mshape

Read-only property returning the shape of the current selection (as computed at construction time for the applicable `select_type`).

### SimpleSelection.tgtshape

Read-only property: for fancy/point selections, returns `list(mshape)`; otherwise returns `[count[dim] for dim in range(rank)]`, i.e. the per-dimension hyperslab counts as a list.

### SimpleSelection.start

Read-only property returning the per-dimension start offsets tuple. Only meaningful when `select_type` is `H5S_SEL_HYPERSLABS`/`H5S_SEL_ALL`.

### SimpleSelection.count

Read-only property returning the per-dimension element counts tuple. Only meaningful for hyperslab/all selections.

### SimpleSelection.step

Read-only property returning the per-dimension step tuple. Only meaningful for hyperslab/all selections.

### SimpleSelection.slices

Read-only property returning a tuple with one entry per dimension, each a `slice`, a `list` of coordinates, or a plain `int` (scalar index). For fancy/point selections this returns the stored per-dimension components directly; for hyperslab/all selections it synthesizes a `slice(start, start + count*step, step)` per dimension from `start`/`count`/`step`.

### SimpleSelection.getSelectNpoints()

Returns the number of selected elements, computed appropriately per `select_type`: `0` for `H5S_SEL_NONE`; the product of `shape` for `H5S_SEL_ALL`; the product of `count` for `H5S_SEL_HYPERSLABS`; and the product of `mshape` for `H5S_SEL_FANCY` (which is correct for both Cartesian-product and paired-coordinate fancy selections).

### SimpleSelection.query_string

Read-only property returning the value of the `select` query parameter used by the HDF REST API for this selection (e.g. `"[0:10,[1,3,5]]"`), or `None` for a scalar (rank-0) shape. For fancy/point selections it renders each dimension's slice as `start:stop[:step]` or its coordinate list as `[c0,c1,...]`; for hyperslab/all selections it renders `start:stop[:step]` per dimension from `start`/`count`/`step`, clamped so `stop` does not exceed the dataspace extent.

### SimpleSelection.broadcast(target_shape)

Generator yielding `(start, count, step, scalar)` tuples for reading `target_shape`-shaped data into this selection's region. Raises `TypeError` for fancy/point selections (unsupported). For a scalar dataspace (`shape == ()`), requires `target_shape` to have exactly one element and yields the selection once. Otherwise it aligns `target_shape` against the selection's `count` from the trailing dimension (NumPy-style broadcasting, treating scalar-indexed axes as size 1), and if the target tiles the selection more than once, yields one offset sub-selection per tile; raises `TypeError` if the shapes are incompatible.

### SimpleSelection.to_dict()

Extends `Selection.to_dict()` with a `"slices"` key when `select_type != H5S_SEL_ALL`: a list with one entry per dimension, each `{"type": "slice", "start": ..., "stop": ..., "step": ...}`, `{"type": "list", "values": [...]}`, or `{"type": "int", "value": ...}`, mirroring `slices`. Reconstructed by the module-level `from_dict()` function.

### SimpleSelection.to_region_json()

Converts the selection to the `{"select_type": ..., "selection": [...]}` shape used to represent HDF5 region references in the h5json format. Only two forms are supported: a paired-coordinate `H5S_SEL_POINTS` selection, which becomes `{"select_type": "H5S_SEL_POINTS", "selection": [[c0, c1, ...], ...]}` (one coordinate list per point, via `_iter_points`); and a unit-step `H5S_SEL_HYPERSLABS`/`H5S_SEL_ALL` selection, which becomes `{"select_type": "H5S_SEL_HYPERSLABS", "selection": [[start, end]]}` where `start`/`end` are per-dimension coordinates and `end` is inclusive (`start[d] + count[d] - 1`). Raises `NotImplementedError` for stepped hyperslabs or `H5S_SEL_FANCY` selections, since neither has an equivalent in the region-reference format. The inverse is the module-level `from_region_json()` function.

## select(obj, args, fields=None)

Factory function that builds a `Selection` from arbitrary indexing arguments. `obj` is either a dataset-like object (with a `.shape` attribute) or a raw shape tuple; `args` is a single argument or tuple of arguments (indices, slices, ellipses, coordinate lists/boolean arrays, a selection-string, a dict, or an existing `Selection`). If `args` is a `dict`, it is interpreted via `_handle_dict_selection` (per-dimension `start`/`stop`/`step`, each broadcastable from a scalar). A string argument is parsed as `"[...]"` (select-all) or via `_getSelectionList` (e.g. `"[0:10,[1,3,5]]"` syntax). If the single argument is already a `Selection` with matching shape, it is returned unchanged (raises `TypeError` on a shape mismatch); a `numpy.ndarray` or `list` is treated as point coordinates (or a boolean mask) via `_points_to_paired`. A scalar object shape (`()`) always returns a `SimpleSelection`. Otherwise returns a new `SimpleSelection(obj_shape, args, fields=fields)`.

## from_query_result(shape, indices)

Builds a `PointSelection`-equivalent (`SimpleSelection` with `H5S_SEL_POINTS`) from an `arrayQuery`-style result: `shape` is the full dataset shape, `indices` is an `(N, rank)` ndarray of selected coordinates. Returns an empty paired selection (via the internal `_empty_paired_sel` helper) if `indices` is empty; for rank-1 shapes delegates to `select()` with a flat coordinate list, otherwise builds a per-dimension coordinate-list tuple.

## intersect(s1, s2)

Returns a new `Selection` representing the intersection of `s1` and `s2`, which must have matching `shape`. Supports hyperslab/hyperslab (clipped per-dimension, raising `ValueError` for stepped slices), hyperslab/fancy or fancy/hyperslab (via the internal `_intersect_fancy_hyperslab` helper, which clips Cartesian-product dimensions independently but filters paired-coordinate dimensions as a unit to keep coordinate lists aligned), and paired-`H5S_SEL_POINTS`/paired-`H5S_SEL_POINTS` (via `_intersect_paired_fancy`, using bounding-box overlap as a fast-reject before an explicit set intersection). The result's `fields` is the intersection of the two inputs' fields (`None` meaning "all fields"). Raises `TypeError` for unsupported type combinations (e.g. two non-paired fancy selections) or `ValueError` for shape mismatches.

## contained(s1, s2)

Returns `True` if every element selected by `s1` is also selected by `s2` (and `s2`'s compound-field set is a subset of `s1`'s), otherwise `False`. Requires matching `shape`. For hyperslab-only combinations it compares `start`/`count` per dimension (conservatively returning `False` for any stepped slice). For combinations involving `H5S_SEL_FANCY`, containment is checked per dimension via the internal `_dim_contained` helper, but returns `False` conservatively whenever either operand is a paired-coordinate selection (more than one list dimension), since containment there isn't a per-dimension (Cartesian) property. Raises `TypeError` for unsupported selection-type combinations or `ValueError` for shape mismatches.

## translate(s1, s2)

Returns a new `Selection`, defined in `s1`'s local coordinate frame, corresponding to the overlap between `s1` and `s2` (`s2` need not be fully contained in `s1` — the intersection is used). Handles three cases: `s1` is fancy/points (intersects in absolute coordinates, then re-expresses each dimension relative to `s1`'s own slice/list/int components); `s2` is fancy/points and `s1` is a plain hyperslab (offsets the intersection by `s1.start`); and both are plain hyperslabs (offsets `s2`'s start by `s1`'s start via `_check_bool_args` validation). Raises `ValueError` if the selections do not overlap, or `TypeError` for unsupported type combinations or shape mismatches.

## from_dict(d)

Reconstructs a `Selection` from a `dict` produced by `Selection.to_dict()`/`SimpleSelection.to_dict()`. Only `"SimpleSelection"` is currently a supported `"class"` value (raises `ValueError` otherwise). If `d["select_type"]` is `H5S_SEL_ALL`, returns a select-all `SimpleSelection`; otherwise rebuilds the per-dimension `slice`/`list`/`int` arguments from `d["slices"]` and constructs `SimpleSelection(shape, args, fields=fields)`.

## from_region_json(d)

Reconstructs a `Selection` from the `{"select_type": ..., "selection": [...]}` dict format used for HDF5 region references in h5json (the inverse of `SimpleSelection.to_region_json()`; see `data/json/regionref_dset.json`). Because this representation does not carry the referenced dataset's true shape, a minimal shape just large enough to contain the given selection is synthesized, so the resulting selection's `.shape` will generally not equal the real dataset's shape. For `"H5S_SEL_POINTS"`, builds a paired-coordinate point selection from the list of coordinate tuples. For `"H5S_SEL_HYPERSLABS"` with a single `[start, end]` block, builds an ordinary hyperslab selection. Since a real HDF5 region reference can contain multiple disjoint hyperslab blocks (e.g. a blocked/strided hyperslab) with no equivalent in this project's `Selection` model, multiple blocks are instead expanded via `itertools.product` into an explicit paired-coordinate point selection covering the exact same cells — noted as potentially memory intensive for selections with very large blocks. Raises `ValueError` for an empty selection, or `NotImplementedError` for any other `select_type`.

## guess_shape(sid)

Given a low-level dataspace/selection identifier `sid` (an h5py-style object exposing `get_simple_extent_type()`, `get_select_type()`, `get_select_npoints()`, `get_select_bounds()`, `shape`, and `copy()`/`select_hyperslab()`), attempts to deduce the shape of the currently active selection. Returns `None` for `H5S_NULL` dataspaces and for an unselected (`H5S_SEL_NONE`) scalar; `()` for a fully-selected scalar; `(0,) * rank` for an empty selection; `sid.shape` for a fully-selected simple dataspace; `(N,)` for a point-based selection (`N` = number of selected points, matching NumPy's flattening of point selections); and for a hyperslab selection, attempts to compute a per-axis extent by masking off each axis in turn and counting leftover points (see the nested `get_n_axis` helper) — falling back to the 1-D shape `(N,)` if the per-axis extents' product doesn't match the true point count (indicating multiple, non-rectangular hyperslab blocks are in effect). Raises `TypeError` for an unrecognized dataspace class or selection method.
