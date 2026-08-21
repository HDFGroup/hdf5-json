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
import itertools
import h5py
from h5py import h5r, h5s, h5t
import numpy as np
from os import stat as os_stat
import time

from ..objid import getCollectionForId, isValidUuid, createObjId
from ..hdf5dtype import getTypeItem, createDataType, isVlen, vlenBaseType, RegionReference, isOpaqueDtype
from ..array_util import bytesArrayToList, jsonToArray
from ..h5py_util import is_reference, is_regionreference, has_reference, convert_dtype
from ..shape_util import getShapeDims, getShapeClass, isExtensible, getMaxDims
from ..track_util import getTrackTimes
from ..dset_util import getDatasetLayout, getFillValue
from ..filters import isCompressionFilter, getFilters, getFilterItem
from .. import selections
from .. import filters
from ..storage_plugin import StoragePlugin


class H5pyPlugin(StoragePlugin):
    """
    This class reads from and writes to a real HDF5 file using h5py.  A single instance holds a
    single h5py.File handle used for both operations, and a single obj_id -> live h5py object map
    (_id_map), so a read always sees whatever this same instance has most recently written -
    including changes made earlier in the same session to objects that existed before it opened.
    """

    def __init__(
        self,
        filepath,
        append=False,
        no_data=False,
        read_only=False,
        app_logger=None
    ):
        super().__init__(filepath, append=append, no_data=no_data, read_only=read_only, app_logger=app_logger)
        self._id_map = {}    # obj_id -> live h5py object (Group/Dataset/Datatype/File for root)
        self._addr_map = {}  # HDF5 object address -> obj_id
        self._init = False if (append or read_only) else True
        self._flush_time = 0.0
        self._f = None  # h5py file handle
        self._root_id = None

    # ------------------------------------------------------------------
    # read-side element/array conversion (h5py -> h5json)
    # ------------------------------------------------------------------

    def _copy_element_in(self, val, src_dt, tgt_dt, fin=None):
        """ convert the given dataset or attribute element from h5py to h5json equivalent """

        out = None
        if len(src_dt) > 0:
            out_fields = []
            i = 0
            for name in src_dt.fields:
                field_src_dt = src_dt.fields[name][0]
                field_tgt_dt = tgt_dt.fields[name][0]
                field_val = val[i]
                i += 1
                out_field = self._copy_element_in(field_val, field_src_dt, field_tgt_dt, fin=fin)
                out_fields.append(out_field)
            out = tuple(out_fields)
        elif src_dt.metadata and "ref" in src_dt.metadata:
            if not tgt_dt.metadata or "ref" not in tgt_dt.metadata:
                raise TypeError(f"Expected tgt dtype to be ref, but got: {tgt_dt}")
            ref = tgt_dt.metadata["ref"]
            if is_reference(ref):
                # initialize out to the h5json-native sentinel for an unset ref
                # (this method converts h5py -> h5json, so a raw h5py.Reference
                # object here would leak an h5py type into an h5json-native array)
                out = "null"

                if ref and val:
                    try:
                        fin_obj = fin[val]
                    except AttributeError as ae:
                        msg = f"Unable able to get obj for ref value: {ae}"
                        self.log.error(msg)
                        raise ValueError(msg)

                    addr = h5py.h5o.get_info(fin_obj.id).addr
                    if addr not in self._addr_map:
                        msg = f"No object found for ref object: {fin_obj.name}"
                        self.log.warning(msg)
                        out = ""
                    else:
                        obj_id = self._addr_map[addr]
                        collection = getCollectionForId(obj_id)
                        out = f"{collection}/{obj_id}"

            elif is_regionreference(ref):
                # We can resolve which dataset a region reference points to
                # (same as for a plain object reference, below), but not
                # what it actually selects within that dataset - so bind the
                # RegionReference to its target dataset only, with no
                # selection.
                out = b''  # null - matches the established "unset" convention

                if val:
                    try:
                        fin_obj = fin[val]
                    except AttributeError as ae:
                        msg = f"Unable able to get obj for ref value: {ae}"
                        self.log.error(msg)
                        raise ValueError(msg)

                    addr = h5py.h5o.get_info(fin_obj.id).addr
                    if addr not in self._addr_map:
                        msg = f"No object found for ref object: {fin_obj.name}"
                        self.log.warning(msg)
                    else:
                        obj_id = self._addr_map[addr]
                        region_ref = RegionReference("datasets/" + obj_id)
                        out = region_ref.tobytes()
            else:
                raise TypeError(f"Unexpected ref type: {type(ref)}")
        elif src_dt.metadata and "vlen" in src_dt.metadata:
            if not isinstance(val, np.ndarray):
                raise TypeError(f"Expecting ndarray or vlen element, but got: {type(val)}")
            if not tgt_dt.metadata or "vlen" not in tgt_dt.metadata:
                raise TypeError(f"Expected tgt dtype to be vlen, but got: {tgt_dt}")
            src_vlen_dt = src_dt.metadata["vlen"]
            tgt_vlen_dt = tgt_dt.metadata["vlen"]
            if has_reference(src_vlen_dt):
                if len(val.shape) == 0:
                    # scalar array
                    e = val[()]
                    v = self._copy_element_in(e, src_vlen_dt, tgt_vlen_dt, fin=fin)
                    out = np.array(v, dtype=tgt_dt)
                else:
                    out = np.zeros(val.shape, dtype=tgt_dt)
                    for i in range(len(out)):
                        e = val[i]
                        out[i] = self._copy_element_in(e, src_vlen_dt, tgt_vlen_dt, fin=fin)
            else:
                # can just directly copy the array
                out = np.zeros(val.shape, dtype=tgt_dt)
                out[...] = val[...]
        else:
            out = val  # can just copy as is
        return out

    def _copy_array_in(self, src_arr, fin=None):
        """Copy the numpy array to a new array.
            Convert any reference type to point to item in the target's hierarchy.
        """

        if not isinstance(src_arr, np.ndarray):
            raise TypeError(f"Expecting ndarray, but got: {src_arr}")
        tgt_dt = convert_dtype(src_arr.dtype, to_h5py=False)
        tgt_arr = np.zeros(src_arr.shape, dtype=tgt_dt)

        if has_reference(src_arr.dtype):
            # flatten array to simplify iteration
            count = int(np.prod(src_arr.shape))
            tgt_arr_flat = tgt_arr.reshape((count,))
            src_arr_flat = src_arr.reshape((count,))
            for i in range(count):
                e = src_arr_flat[i]
                element = self._copy_element_in(e, src_arr.dtype, tgt_dt, fin=fin)
                tgt_arr_flat[i] = element
            tgt_arr = tgt_arr_flat.reshape(src_arr.shape)
        else:
            # can just copy the entire array
            tgt_arr[...] = src_arr[...]
        return tgt_arr

    # ------------------------------------------------------------------
    # write-side element/array conversion (h5json -> h5py)
    # ------------------------------------------------------------------

    def _buildRegionDataspace(self, target_obj, region_ref):
        """ Build an h5py low-level dataspace (h5py.h5s.SpaceID) with
        region_ref's selection applied against target_obj, suitable for
        passing to h5r.create() to make a real HDF5 region reference.

        The low-level H5S selection API is used directly (rather than
        target_obj.regionref[...]) since h5py's high-level slicing only
        supports a single fancy-index array per call, and can't express a
        paired-coordinate point selection (H5S_SEL_POINTS) at all.
        """
        sid = target_obj.id.get_space()

        if region_ref.selection_bytes is None:
            # no selection bound - reference the whole dataset
            sid.select_all()
            return sid

        sel = selections.Selection.frombytes(region_ref.selection_bytes)
        rank = len(sel.shape)

        if rank == 0:
            sid.select_all()
            return sid

        def dim_start_count_step(s):
            if isinstance(s, slice):
                return s.start, (s.stop - s.start) // s.step, s.step
            return int(s), 1, 1

        if sel.select_type == selections.H5S_SEL_POINTS:
            points = list(selections._iter_points(sel))
            sid.select_elements(points)
            return sid

        if sel.select_type in (selections.H5S_SEL_HYPERSLABS, selections.H5S_SEL_ALL):
            starts, counts, steps = zip(*(dim_start_count_step(s) for s in sel.slices))
            sid.select_hyperslab(starts, counts, stride=steps)
            return sid

        if sel.select_type == selections.H5S_SEL_FANCY:
            # a mix of slices/ints and coordinate lists (Cartesian product) -
            # HDF5 has no single primitive for this, so union one hyperslab
            # block per combination of list-dim values (matching how HDF5
            # itself decomposes e.g. dset[[0, 2], 1:4] internally)
            slices = sel.slices
            list_dims = [d for d in range(rank) if isinstance(slices[d], list)]
            value_choices = [slices[d] for d in list_dims]
            for i, combo in enumerate(itertools.product(*value_choices)):
                combo_map = dict(zip(list_dims, combo))
                starts, counts, steps = [], [], []
                for d in range(rank):
                    if d in combo_map:
                        starts.append(int(combo_map[d]))
                        counts.append(1)
                        steps.append(1)
                    else:
                        start, count, step = dim_start_count_step(slices[d])
                        starts.append(start)
                        counts.append(count)
                        steps.append(step)
                op = h5s.SELECT_SET if i == 0 else h5s.SELECT_OR
                sid.select_hyperslab(tuple(starts), tuple(counts), stride=tuple(steps), op=op)
            return sid

        raise NotImplementedError(f"Cannot create HDF5 region reference for select_type {sel.select_type}")

    def _copy_element_out(self, val, src_dt, tgt_dt, fout=None):
        """ convert the given dataset or attribute element to h5py equivalent """
        out = None
        if len(src_dt) > 0:
            out_fields = []
            i = 0
            for name in src_dt.fields:
                field_src_dt = src_dt.fields[name][0]
                field_tgt_dt = tgt_dt.fields[name][0]
                field_val = val[i]
                i += 1
                out_field = self._copy_element_out(field_val, field_src_dt, field_tgt_dt)
                out_fields.append(out_field)
            out = tuple(out_fields)
        elif src_dt.metadata and "ref" in src_dt.metadata:
            if not tgt_dt.metadata or "ref" not in tgt_dt.metadata:
                raise TypeError(f"Expected tgt dtype to be ref, but got: {tgt_dt}")
            ref = tgt_dt.metadata["ref"]
            if is_reference(ref):
                # initialize out to null ref
                out = h5py.Reference()  # null h5py ref

                if val:
                    if isinstance(val, bytes):
                        val = val.decode("ascii")
                    if val == "null":
                        pass  # on-the-wire sentinel for an unset reference - leave out as null ref
                    else:
                        # strip out collection prefix if present
                        parts = val.split("/")
                        obj_uuid = parts[-1]
                        if not isValidUuid(obj_uuid):
                            msg = f"invalid uuid: {obj_uuid}"
                            self.log.warning(msg)
                        elif obj_uuid not in self._id_map:
                            self.log.warning(f"ref object {obj_uuid} not found")
                        else:
                            out = self._id_map[obj_uuid].ref

            elif is_regionreference(ref):
                # initialize out to null region ref
                out = h5py.RegionReference()

                raw = val.item() if isinstance(val, np.ndarray) else val
                if raw:
                    region_ref = raw if isinstance(raw, RegionReference) else RegionReference.frombytes(raw)
                    if region_ref.id is not None:
                        obj_id = region_ref.id
                        if obj_id not in self._id_map:
                            self.log.warning(f"region ref object {obj_id} not found")
                        else:
                            target_obj = self._id_map[obj_id]
                            if not isinstance(target_obj, h5py.Dataset):
                                self.log.warning(f"region ref target {obj_id} is not a dataset")
                            else:
                                try:
                                    sid = self._buildRegionDataspace(target_obj, region_ref)
                                    out = h5r.create(target_obj.id, b'.', h5r.DATASET_REGION, sid)
                                except (NotImplementedError, ValueError) as e:
                                    self.log.warning(f"unable to create region reference: {e}")
            else:
                raise TypeError(f"Unexpected ref type: {type(ref)}")
        elif src_dt.metadata and "vlen" in src_dt.metadata:
            if not tgt_dt.metadata or "vlen" not in tgt_dt.metadata:
                raise TypeError(f"Expected tgt dtype to be vlen, but got: {tgt_dt}")
            src_vlen_dt = src_dt.metadata["vlen"]
            tgt_vlen_dt = tgt_dt.metadata["vlen"]

            if has_reference(src_vlen_dt):
                if isinstance(val, np.ndarray) and val.shape == ():
                    val = val[()]
                if isinstance(val, np.ndarray) or isinstance(val, list) or isinstance(val, tuple):
                    count = len(val)
                    out = np.zeros((count,), dtype=tgt_dt)
                    for i in range(count):
                        e = val[i]
                        out[i] = self._copy_element_out(e, src_vlen_dt, tgt_vlen_dt, fout=fout)
                else:
                    # scalar array
                    v = self._copy_element_out(val, src_vlen_dt, tgt_vlen_dt, fout=fout)
                    out = np.array(v, dtype=tgt_dt)
            else:
                # can just directly copy the array
                out = np.zeros(val.shape, dtype=tgt_dt)
                out[...] = val[...]
        else:
            out = val  # can just copy as is
        return out

    def _writeDatasetFull(self, dset, arr):
        """ write arr as the full contents of dset (arr.shape == dset.shape). """
        self._writeDatasetRegion(dset, arr)

    def _writeDatasetRegion(self, dset, arr, slices=None):
        """ write arr into dset, either as the full contents (slices=None) or a hyperslab region
        (slices - a tuple of slice objects, one per dimension, matching arr.shape).
        dset[...] = arr / dset[slices] = arr can mis-handle a numpy object-dtype array (used for
        vlen/RegionReference data) when every element happens to have the same length - e.g.
        all-empty vlen arrays - since numpy "helpfully" homogenizes it into a plain N-d array
        during the high-level write, and h5py then rejects the now-wrong shape
        ("Can't broadcast (4, 0) -> (4,)"). write_direct bypasses that broadcasting/reshaping
        logic, but can't be used unconditionally: for an H5T_ARRAY (subarray) dtype, arr's actual
        shape includes the subarray dims (e.g. (4, 3, 5) for a (4,) dataset of (3, 5) arrays),
        which only the high-level path knows how to reconcile against dset.shape/the given
        slices. """
        if arr.dtype.kind == "O":
            dset.write_direct(arr, dest_sel=slices)
        elif slices is None:
            dset[...] = arr
        else:
            dset[slices] = arr

    def _copy_array_out(self, src_arr, fout=None):
        """Copy the numpy array to a new array.
            Convert any reference type to point to item in the target's hierarchy.
        """
        if not isinstance(src_arr, np.ndarray):
            raise TypeError(f"Expecting ndarray, but got: {src_arr}")
        tgt_dt = convert_dtype(src_arr.dtype, to_h5py=True)
        tgt_arr = np.zeros(src_arr.shape, dtype=tgt_dt)

        if has_reference(src_arr.dtype):
            # flatten array to simplify iteration
            count = int(np.prod(src_arr.shape))
            tgt_arr_flat = tgt_arr.reshape((count,))
            src_arr_flat = src_arr.reshape((count,))
            for i in range(count):
                e = src_arr_flat[i]
                element = self._copy_element_out(e, src_arr.dtype, tgt_dt, fout=fout)
                tgt_arr_flat[i] = element
            tgt_arr = tgt_arr_flat.reshape(src_arr.shape)
        elif len(src_arr.dtype) == 0 and isVlen(src_arr.dtype) and vlenBaseType(src_arr.dtype) in (bytes, str):
            # vlen strings need elements converted to Python str for h5py
            count = int(np.prod(src_arr.shape))
            tgt_dt = h5py.special_dtype(vlen=str)
            tgt_arr = np.zeros(src_arr.shape, dtype=tgt_dt)
            tgt_arr_flat = tgt_arr.reshape((count,))
            src_arr_flat = src_arr.reshape((count,))
            for i in range(count):
                e = src_arr_flat[i]
                if isinstance(e, str):
                    tgt_arr_flat[i] = e
                elif isinstance(e, bytes):
                    tgt_arr_flat[i] = e.decode('utf-8')
                elif isinstance(e, np.ndarray) and e.dtype.kind == 'S':
                    # numpy byte string array - convert to Python string
                    tgt_arr_flat[i] = e.item().decode('utf-8')
                elif isinstance(e, np.ndarray) and e.dtype.kind == 'U':
                    # numpy unicode array - get Python string
                    tgt_arr_flat[i] = e.item()
                elif isinstance(e, np.bytes_):
                    tgt_arr_flat[i] = e.decode('utf-8')
                elif isinstance(e, np.str_):
                    tgt_arr_flat[i] = str(e)
                else:
                    tgt_arr_flat[i] = e
            tgt_arr = tgt_arr_flat.reshape(src_arr.shape)
        else:
            # can just copy the entire array
            tgt_arr[...] = src_arr[...]
        return tgt_arr

    def _createGroup(self, parent, grp_json, name=None):
        """ create the group and any links it contains """
        cpl = grp_json.get("creationProperties") or {}
        track_times = getTrackTimes(grp_json)
        link_creation_order = cpl.get("linkCreationOrder")

        if track_times is None and link_creation_order is None:
            # no relevant group-level creation properties - the common case
            return parent.create_group(name)

        # h5py's high-level create_group() only exposes a single track_order
        # bool, which sets both CRT_ORDER_TRACKED and CRT_ORDER_INDEXED
        # together - there's no way to request "tracked but not indexed"
        # (H5P_CRT_ORDER_TRACKED alone) through it, so build the GCPL
        # directly via the low-level API whenever either property is set
        encoded_name, lcpl = parent._e(name, lcpl=True)
        gcpl = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
        if track_times is not None:
            gcpl.set_obj_track_times(track_times)
        if link_creation_order == "H5P_CRT_ORDER_TRACKED":
            gcpl.set_link_creation_order(h5py.h5p.CRT_ORDER_TRACKED)
        elif link_creation_order == "H5P_CRT_ORDER_INDEXED":
            gcpl.set_link_creation_order(h5py.h5p.CRT_ORDER_TRACKED | h5py.h5p.CRT_ORDER_INDEXED)
        elif link_creation_order is not None:
            raise ValueError(f"unexpected linkCreationOrder: {link_creation_order}")

        gid = h5py.h5g.create(parent.id, encoded_name, lcpl=lcpl, gcpl=gcpl)
        return h5py.Group(gid)

    def _createDataset(self, parent, dset_json, name=None):
        """ create a dataset object """

        dtype = self.db.getDtype(dset_json)
        # h5py's type layer identity-checks special (ref/vlen) dtype metadata
        # against its own Reference/RegionReference classes, so the h5json
        # dtype must be translated before being handed to create_dataset() -
        # everything else in this method keeps using the untranslated dtype,
        # since e.g. jsonToArray() needs h5json's own metadata to recognize it.
        h5py_dtype = convert_dtype(dtype, to_h5py=True)

        kwargs = {"dtype": h5py_dtype}
        shape_class = getShapeClass(dset_json)
        if shape_class == "H5S_NULL":
            # skip the shape keyword to create a null space dataset
            pass
        elif shape_class == "H5S_SCALAR":
            kwargs["shape"] = ()
        else:
            shape = getShapeDims(dset_json)
            kwargs["shape"] = shape
            if isExtensible(dset_json):
                maxshape = list(getMaxDims(dset_json))
                # replace any 0, or H5S_UNLIMITED with None
                for dim in range(len(maxshape)):
                    if maxshape[dim] in (0, "H5S_UNLIMITED"):
                        maxshape[dim] = None
                kwargs["maxshape"] = tuple(maxshape)

        fillvalue = getFillValue(dset_json)

        if fillvalue and len(dtype) > 1 and type(fillvalue) in (list, tuple):
            # for compound types, need to convert from list to dataset compatible element

            if len(dtype) != len(fillvalue):
                msg = "fillvalue has incorrect number of elements"
                raise ValueError(msg)

            fillvalue = jsonToArray((), dtype, fillvalue)

        kwargs["fillvalue"] = fillvalue

        track_times = getTrackTimes(dset_json)
        if track_times is not None:
            kwargs["track_times"] = track_times

        layout = getDatasetLayout(dset_json)
        if layout and "dims" in layout:
            kwargs["chunks"] = tuple(layout["dims"])

        filter_props = getFilters(dset_json)

        for filter_prop in filter_props:
            try:
                getFilterItem(filter_prop)
            except (KeyError, ValueError, TypeError):
                self.log.warning(f"unknown filter: {filter_prop} ignoring")
                continue
            filter_class = filter_prop["class"]
            filter_id = filter_prop["id"]
            filter_name = filter_prop["name"]

            if not h5py.h5z.filter_avail(filter_id):
                msg = f"filter not available, filter: {filter_class}, ignoring"
                self.log.warning(msg)
                continue

            if isCompressionFilter(filter_class):
                if kwargs.get("compression"):
                    msg = f"compression filter already set for {filter_class}, ignoring"
                    self.log.info(msg)
                    continue

                kwargs["compression"] = filter_name
                self.log.info(f"setting compression filter to: {filter_class}")
                if filter_class == "H5Z_FILTER_DEFLATE":
                    kwargs["compression"] = "gzip"  # h5py doesn't recognize 'deflate' name
                    # check for an optional compression value
                    if "level" in filter_prop:
                        kwargs["compression_opts"] = filter_prop["level"]
                elif filter_class == "H5Z_FILTER_SZIP":
                    bitsPerPixel = None
                    coding = "nn"

                    if "bitsPerPixel" in filter_prop:
                        bitsPerPixel = filter_prop["bitsPerPixel"]
                    if "coding" in filter_prop:
                        if filter_prop["coding"] == "H5_SZIP_EC_OPTION_MASK":
                            coding = "ec"
                        elif filter_prop["coding"] == "H5_SZIP_NN_OPTION_MASK":
                            coding = "nn"
                        else:
                            self.log.warning("invalid szip option: 'coding'")
                        # note: pixelsPerBlock, and pixelsPerScanline not supported by h5py,
                        # so these options will be ignored
                    if "pixelsPerBlock" in filter_props:
                        self.log.info("ignoring szip option: 'pixelsPerBlock'")
                    if "pixelsPerScanline" in filter_props:
                        self.log.info("ignoring szip option: 'pixelsPerScanline'")
                    if bitsPerPixel:
                        kwargs["compression_opts"] = (coding, bitsPerPixel)
                elif filter_class == "H5Z_FILTER_SHUFFLE":
                    kwargs["shuffle"] = True
                elif filter_class == "H5Z_FILTER_FLETCHER32":
                    kwargs["fletcher32"] = True
                elif filter_class == "H5Z_FILTER_SCALEOFFSET":
                    if "scaleOffset" in filter_prop:
                        kwargs["scaleoffset"] = filter_prop["scaleOffset"]
                else:
                    self.log.warning(f"Ignoring filter: {filter_class}")

        dset = parent.create_dataset(name, **kwargs)
        return dset

    def _createDatatype(self, parent, ctype_json, name=None):
        """ create a datatype object """

        type_item = ctype_json["type"]
        dtype = createDataType(type_item)
        parent[name] = dtype
        return parent[name]

    def _createObjects(self, parent, links_json, visited=set()):
        """ create child object in the given group, recurse for any sub-groups """

        titles = list(links_json.keys())
        for title in titles:
            link_json = links_json[title]
            link_class = link_json["class"]
            if "DELETED" in link_json:
                if title in parent:
                    # delete the link
                    self.log.debug(f"deleting link {title}")
                    del parent[title]
                # update the link json
                del links_json[title]
                continue

            if link_class == "H5L_TYPE_SOFT" and title not in parent:
                h5path = link_json["h5path"]
                parent[title] = h5py.SoftLink(h5path)
            elif link_class == "H5L_TYPE_EXTERNAL" and title not in parent:
                h5path = link_json["h5path"]
                filename = link_json["file"]
                parent[title] = h5py.ExternalLink(filename, h5path)
            elif link_class == "H5L_TYPE_USER_DEFINED" and title not in parent:
                self.log.warning(f"unable to create user-defined link: {title}")
            elif link_class == "H5L_TYPE_HARD":
                tgt_id = link_json["id"]

                collection = getCollectionForId(tgt_id)

                obj_json = self.db.getObjectById(tgt_id)

                if tgt_id in self._id_map:
                    # object has already been created
                    tgt_obj = self._id_map[tgt_id]
                    if title not in parent:
                        parent[title] = tgt_obj
                    if collection == "groups" and tgt_id not in visited:
                        # recurse over sub-objects to pick up any new links
                        grp_links = obj_json["links"]
                        visited.add(tgt_id)
                        self._createObjects(tgt_obj, grp_links, visited=visited)
                elif title in parent:
                    # the link already exists on disk from an earlier session
                    # (e.g. this plugin was closed and reopened in append
                    # mode) - _id_map was cleared on close(), so reconnect
                    # tgt_id to the existing object instead of trying to
                    # create a duplicate, which h5py would reject
                    tgt_obj = parent[title]
                    self._addCreatedObject(tgt_id, tgt_obj)
                    if collection == "groups" and tgt_id not in visited and "links" in obj_json:
                        grp_links = obj_json["links"]
                        visited.add(tgt_id)
                        self._createObjects(tgt_obj, grp_links, visited=visited)
                else:
                    # need to create tgt_id object
                    kwds = {"name": title}
                    if collection == "groups":
                        tgt_grp = self._createGroup(parent, obj_json, **kwds)
                        self._addCreatedObject(tgt_id, tgt_grp)
                        if "links" in obj_json:
                            grp_links = obj_json["links"]
                            visited.add(tgt_id)
                            self._createObjects(tgt_grp, grp_links, visited=visited)
                    elif collection == "datasets":
                        tgt_dset = self._createDataset(parent, obj_json, **kwds)
                        self._addCreatedObject(tgt_id, tgt_dset)
                    elif collection == "datatypes":
                        tgt_ctype = self._createDatatype(parent, obj_json, **kwds)
                        self._addCreatedObject(tgt_id, tgt_ctype)
                    else:
                        self.log.warning(f"unexpected collection: {collection}")
                visited.add(tgt_id)

            else:
                self.log.warning(f"unexpected link class: {link_class}")

    def _addCreatedObject(self, obj_id, obj):
        """ record a newly-created object in the shared id/address maps, so
        later reads (in this session or a future one, e.g. via copy()) and
        reference resolution can find it the same way as any pre-existing
        object discovered by _getHardLinkIds(). """
        self._id_map[obj_id] = obj
        addr = h5py.h5o.get_info(obj.id).addr
        self._addr_map[addr] = obj_id

    def resizeDataset(self, dset_id, dset):
        """ Update the datasets shape """

        dset_json = self.db.getObjectById(dset_id)
        new_dims = getShapeDims(dset_json)
        dset.resize(new_dims)

    def updateDatasetValues(self, dset_id, dset):
        """ write any pending dataset values """

        updates = self.db._getDatasetUpdates(dset_id)

        for (sel, val) in updates:
            if val is not None and has_reference(val.dtype):
                # val is still in h5json-native form (e.g. "S48" id strings for
                # H5T_STD_REF_OBJ) - h5py's dataset write path expects actual
                # h5py.Reference/RegionReference elements, so it has to be
                # converted before being handed to any of the branches below
                val = self._copy_array_out(val, fout=dset.file)
            if sel is None or sel.select_type == selections.H5S_SEL_NONE:
                pass  # no updates
            elif sel.select_type == selections.H5S_SEL_ALL:
                self._writeDatasetFull(dset, val)
                self.log.debug(f"h5py_plugin dset {dset.name} updated with sel_all")
            elif sel.select_type == selections.H5S_SEL_HYPERSLABS:
                slices = []
                for dim in range(len(sel.shape)):
                    start = sel.start[dim]
                    stop = start + sel.count[dim]
                    step = sel.step[dim]
                    slices.append(slice(start, stop, step))
                slices = tuple(slices)
                self._writeDatasetRegion(dset, val, slices=slices)
            elif sel.select_type == selections.H5S_SEL_POINTS:
                for i, pt in enumerate(selections._iter_points(sel)):
                    dset[pt] = val[i]
                self.log.debug(f"h5py_plugin dset {dset.name} updated with point selection")
            elif sel.select_type == selections.H5S_SEL_FANCY:
                rank = len(sel.shape)
                slices = sel.slices
                list_dims = [d for d in range(rank) if isinstance(slices[d], list)]
                if len(list_dims) > 1:
                    # Multiple coordinate lists: decompose into n per-pair writes.
                    list_dims_set = set(list_dims)
                    n = len(slices[list_dims[0]])
                    for i in range(n):
                        idx = tuple(
                            int(slices[d][i]) if d in list_dims_set else slices[d]
                            for d in range(rank)
                        )
                        dset[idx] = val[i]
                else:
                    dset[slices] = val
                self.log.debug(f"h5py_plugin dset {dset.name} updated with fancy selection")
            else:
                raise TypeError(f"Unexpected selection type: {type(sel)}")

            self.log.debug(f"h5py_plugin dset {dset.name} updated")

    def createAttribute(self, obj, name, attr_json):
        """ add the given attribute to obj """

        src_dt = self.db.getDtype(attr_json)

        # handle special case of null space attribute here
        shape_json = attr_json["shape"]
        shape_class = shape_json["class"]
        if shape_class == "H5S_NULL":
            obj.attrs[name] = h5py.Empty(convert_dtype(src_dt, to_h5py=True))
            return

        if shape_class == "H5S_SCALAR":
            dims = ()
        else:
            dims = shape_json["dims"]
        src_arr = jsonToArray(dims, src_dt, attr_json["value"])
        if not isinstance(src_arr, np.ndarray):
            raise TypeError("Unexpected type for src_arr")
        tgt_arr = self._copy_array_out(src_arr, fout=obj.file)
        # obj.attrs[name] = tgt_arr can mis-handle a numpy object-dtype array
        # (vlen/RegionReference data) when every element happens to have the
        # same length - e.g. all-empty vlen arrays - since numpy "helpfully"
        # homogenizes it into a plain N-d array during the write. Passing an
        # explicit dtype through attrs.create() preserves tgt_arr's actual
        # dtype (e.g. vlen-of-reference) instead of letting h5py re-infer it
        # from the array's runtime contents.
        obj.attrs.create(name, data=tgt_arr, dtype=tgt_arr.dtype)

    def updateAttributes(self, obj_id, obj):
        """ create/replace any modified attributes """

        obj_json = self.db.getObjectById(obj_id)

        if "attributes" not in obj_json:
            # no attributes
            return

        attrs = obj_json["attributes"]
        for name in attrs:
            attr_json = attrs[name]
            if "DELETED" in attr_json:
                if name in obj.attrs:
                    # delete the attribute
                    self.log.debug(f"h5py_plugin - delete attribute {name}")
                    del obj.attrs[name]
                else:
                    pass  # already deleted or never added
                continue
            if "created" in attr_json and attr_json["created"] < self._flush_time:
                # attribute should be saved already
                continue
            self.createAttribute(obj, name, attr_json)

    # ------------------------------------------------------------------
    # read-side object/attribute/link retrieval
    # ------------------------------------------------------------------

    def _readOpaqueAttribute(self, attrObj):
        """ Read the full opaque attribute via the low-level API using a
        memory type tagged to match the file type - see
        _readOpaqueDataset() for why this is needed. """
        file_tid = attrObj.get_type()
        itemsize = file_tid.get_size()
        mem_tid = h5t.create(h5t.OPAQUE, itemsize)
        tag = file_tid.get_tag()
        if tag:
            mem_tid.set_tag(tag)
        buf = np.zeros(attrObj.shape, dtype=f"V{itemsize}")
        attrObj.read(buf, mtype=mem_tid)
        return buf

    def getAttribute(self, obj_id, name, include_data=True):
        """ Return JSON for the given attribute """

        obj = self._id_map[obj_id]

        if name not in obj.attrs:
            msg = f"Attribute: [{name}] not found in object: {obj.name}"
            self.log.info(msg)
            return None

        # get the attribute!
        attrObj = h5py.h5a.open(obj.id, np.bytes_(name))

        item = {}

        # check if the dataset is using a committed type
        typeid = attrObj.get_type()
        type_item = None
        if h5py.h5t.TypeID.committed(typeid):
            type_uuid = None
            addr = h5py.h5o.get_info(typeid).addr
            type_uuid = self.getObjIdByAddress(addr)
            committedType = self._id_map[type_uuid]
            type_item = getTypeItem(committedType.dtype)
            type_item["id"] = type_uuid
        else:
            type_item = getTypeItem(attrObj.dtype)
        item["type"] = type_item

        shape_item = {}
        if attrObj.shape is None or attrObj.get_storage_size() == 0:
            # If storage size is 0, assume this is a null space obj
            # See: h5py issue https://github.com/h5py/h5py/issues/279
            shape_item["class"] = "H5S_NULL"
        else:
            if attrObj.shape:
                shape_item["class"] = "H5S_SIMPLE"
                shape_item["dims"] = attrObj.shape
            else:
                shape_item["class"] = "H5S_SCALAR"

        is_opaque = isinstance(type_item, dict) and type_item["class"] == "H5T_OPAQUE"

        item["shape"] = shape_item
        if shape_item["class"] == "H5S_NULL":
            include_data = False
        else:
            pass  # use include_data parameter

        if include_data:
            try:
                if is_opaque:
                    # h5py's high-level attrs[] requires the memory type's
                    # tag to match the file type's (real HDF5 opaque data
                    # usually has one) - read via the low-level API instead
                    data = self._readOpaqueAttribute(attrObj)
                else:
                    data = obj.attrs[name]
                # convert from h5py to h5json
                data = self._copy_array_in(data, fin=obj.file)
            except TypeError:
                self.log.warning("type error reading attribute")

        if include_data and data is not None:
            value = bytesArrayToList(data)
            item["value"] = value
            if is_opaque:
                item["encoding"] = "base64"
        else:
            pass  # no data
        stats = self.getStats()
        item['created'] = stats["lastModified"]  # use file modification time as attr creation time
        return item

    def getAttributes(self, obj_id, include_data=True):
        h5obj = self._id_map[obj_id]
        self.log.info(f"getAttributes: {obj_id} include_data={include_data}")
        items = {}  # with python 3.7+, this will maintain the attribute order we got from h5py
        attrs = h5obj.attrs
        for name in attrs:
            item = self.getAttribute(obj_id, name, include_data=include_data)
            items[name] = item

        return items

    def _getLink(self, parent, link_name):
        if link_name not in parent:
            return None

        item = {"title": link_name}
        # get the link object, one of HardLink, SoftLink, or ExternalLink
        try:
            linkObj = parent.get(link_name, None, False, True)
            linkClass = linkObj.__class__.__name__
        except TypeError:
            # UDLink? set class as 'user'
            linkClass = "UDLink"  # user defined links
            item["class"] = "H5L_TYPE_USER_DEFINED"
        if linkClass == "SoftLink":
            item["class"] = "H5L_TYPE_SOFT"
            item["h5path"] = linkObj.path
        elif linkClass == "ExternalLink":
            item["class"] = "H5L_TYPE_EXTERNAL"
            item["h5path"] = linkObj.path
            item["file"] = linkObj.filename
        elif linkClass == "HardLink":
            # Hardlink doesn't have any properties itself, just get the linked
            # object
            obj = parent[link_name]
            addr = h5py.h5o.get_info(obj.id).addr
            item["class"] = "H5L_TYPE_HARD"
            if addr not in self._addr_map:
                self.log.error(f"expected to find addr for link {link_name} in addr_map")
                item["id"] = None
            else:
                item["id"] = self._addr_map[addr]

        stats = self.getStats()
        item['created'] = stats["lastModified"]  # use file modification time as attr creation time

        return item

    def _getLinks(self, grp):
        items = {}  # with python 3.7+, this will maintain the link order we got from h5py
        for link_name in grp:
            item = self._getLink(grp, link_name)
            items[link_name] = item
        return items

    def _getGroup(self, grp, include_links=True):
        self.log.info(f"_getGroup alias: [{grp.name}]")

        item = {"alias": grp.name}

        # link creation order (unlike track_times) is reliably queryable back
        # from an existing group's GCPL, so report it when set
        order_flags = grp.id.get_create_plist().get_link_creation_order()
        if order_flags & h5py.h5p.CRT_ORDER_INDEXED:
            item["creationProperties"] = {"linkCreationOrder": "H5P_CRT_ORDER_INDEXED"}
        elif order_flags & h5py.h5p.CRT_ORDER_TRACKED:
            item["creationProperties"] = {"linkCreationOrder": "H5P_CRT_ORDER_TRACKED"}

        if include_links:
            links = self._getLinks(grp)
            item["links"] = links
        return item

    def _getDatatype(self, ctype):
        self.log.info(f"getDatatype alias: ]{ctype.name}")
        item = {"alias": ctype.name}
        item["type"] = getTypeItem(ctype.dtype)

        return item

    def _getHDF5DatasetCreationProperties(self, dset):
        """ Get dataset creation properties maintained by HDF5 library """

        #
        # Fill in creation properties
        #
        creationProps = {}
        plist = h5py.h5d.DatasetID.get_create_plist(dset.id)

        # alloc time
        nAllocTime = plist.get_alloc_time()
        if nAllocTime == h5py.h5d.ALLOC_TIME_DEFAULT:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_DEFAULT"
        elif nAllocTime == h5py.h5d.ALLOC_TIME_LATE:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_LATE"
        elif nAllocTime == h5py.h5d.ALLOC_TIME_EARLY:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_EARLY"
        elif nAllocTime == h5py.h5d.ALLOC_TIME_INCR:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_INCR"
        else:
            self.log.warning(f"Unknown alloc time value: {nAllocTime}")

        # fill time
        nFillTime = plist.get_fill_time()
        if nFillTime == h5py.h5d.FILL_TIME_ALLOC:
            creationProps["fillTime"] = "H5D_FILL_TIME_ALLOC"
        elif nFillTime == h5py.h5d.FILL_TIME_NEVER:
            creationProps["fillTime"] = "H5D_FILL_TIME_NEVER"
        elif nFillTime == h5py.h5d.FILL_TIME_IFSET:
            creationProps["fillTime"] = "H5D_FILL_TIME_IFSET"
        else:
            self.log.warning(f"unknown fill time value: {nFillTime}")

        if plist.fill_value_defined() == h5py.h5d.FILL_VALUE_USER_DEFINED:
            creationProps["fillValue"] = bytesArrayToList(dset.fillvalue)

        # layout
        nLayout = plist.get_layout()
        if nLayout == h5py.h5d.COMPACT:
            creationProps["layout"] = {"class": "H5D_COMPACT"}
        elif nLayout == h5py.h5d.CONTIGUOUS:
            creationProps["layout"] = {"class": "H5D_CONTIGUOUS"}
        elif nLayout == h5py.h5d.CHUNKED:
            creationProps["layout"] = {"class": "H5D_CHUNKED", "dims": dset.chunks}
        else:
            self.log.warning(f"Unknown layout value: {nLayout}")

        num_filters = plist.get_nfilters()
        filter_props = []
        if num_filters:
            for n in range(num_filters):
                filter_info = plist.get_filter(n)
                opt_values = filter_info[2]
                filter_prop = {}
                filter_id = filter_info[0]
                filter_prop["id"] = filter_id
                if filter_info[3]:
                    filter_prop["name"] = bytesArrayToList(filter_info[3])
                hdf_filter = filters.getFilterItem(filter_id)
                if hdf_filter:
                    filter_prop["class"] = hdf_filter["class"]
                    if "options" in hdf_filter:
                        filter_opts = hdf_filter["options"]
                        for i in range(len(filter_opts)):
                            if len(opt_values) <= i:
                                break  # end of option values
                            opt_value = opt_values[i]
                            opt_value_enum = None
                            option_name = filter_opts[i]
                            if option_name in filters.HDF_FILTER_OPTION_ENUMS:
                                option_enums = filters.HDF_FILTER_OPTION_ENUMS[option_name]
                                if opt_value in option_enums:
                                    opt_value_enum = option_enums[opt_value]
                            if opt_value_enum:
                                filter_prop[option_name] = opt_value_enum
                            else:
                                filter_prop[option_name] = opt_value
                else:
                    # custom filter
                    filter_prop["class"] = "H5Z_FILTER_USER"
                    if opt_values:
                        filter_prop["parameters"] = opt_values
                filter_props.append(filter_prop)
            creationProps["filters"] = filter_props

        return creationProps

    def _getDataset(self, dset):
        """ return json representation of the given dataset """

        self.log.info(f"getDataset alias: [{dset.name}]")

        item = {"alias": dset.name}
        typeid = dset.id.get_type()
        if h5py.h5t.TypeID.committed(typeid):
            type_uuid = None
            addr = h5py.h5o.get_info(typeid).addr
            type_uuid = self.getObjIdByAddress(addr)
            committedType = self.getObjectById(type_uuid)
            type_item = committedType["type"]
            type_item["id"] = type_uuid
        else:
            type_item = getTypeItem(dset.dtype)
        item["type"] = type_item

        shape_item = {}
        if dset.shape is None:
            # new with h5py 2.6, null space datasets will return None for shape
            shape_item["class"] = "H5S_NULL"
        elif len(dset.shape) == 0:
            shape_item["class"] = "H5S_SCALAR"
        else:
            shape_item["class"] = "H5S_SIMPLE"
            shape_item["dims"] = list(dset.shape)
            maxshape = []
            include_maxdims = False
            for i in range(len(dset.shape)):
                extent = 0
                if len(dset.maxshape) > i:
                    extent = dset.maxshape[i]
                    if extent is None:
                        extent = 0
                    if extent > dset.shape[i] or extent == 0:
                        include_maxdims = True
                maxshape.append(extent)
            if include_maxdims:
                shape_item["maxdims"] = maxshape
        item["shape"] = shape_item

        item["creationProperties"] = self._getHDF5DatasetCreationProperties(dset)

        return item

    def _getHardLinkIds(self, parent):
        """ create any ids for hard links of the group """

        self.log.debug(f"h5py_plugin> _getHardlinkIds for {parent.name}")
        for link_name in parent:
            self.log.debug(f"h5py_plugin> check link: {link_name}")

            try:
                linkObj = parent.get(link_name, None, False, True)
                linkClass = linkObj.__class__.__name__
            except TypeError:
                # UDLink? Go on to the next link
                continue
            if linkClass != "HardLink":
                self.log.debug(f"h5py_plugin> ignoring {link_name} - type: {linkClass}")
            else:
                # get the linked object
                obj = parent[link_name]
                addr = h5py.h5o.get_info(obj.id).addr
                if addr not in self._addr_map:
                    name = obj.__class__.__name__
                    obj_id = createObjId(obj_type=name, root_id=self._root_id)  # create uuid
                    self.log.debug(f"h5py_plugin> creating obj_id: {obj_id} for obj: {obj.name}")
                    self._id_map[obj_id] = obj
                    self._addr_map[addr] = obj_id
                else:
                    obj_id = self._addr_map[addr]
                    if obj_id not in self._id_map:
                        self.log.debug(f"h5py_plugin> adding obj for {obj_id} to id_map")
                        self._id_map[obj_id] = obj
                    else:
                        self.log.debug(f"h5py_plugin> obj {obj_id} already in id_map")

    def getObjectById(self, obj_id, include_attrs=True, include_links=True):
        """ return object with given id """
        if obj_id not in self._id_map:
            raise KeyError(f"{obj_id} not found")
        h5obj = self._id_map[obj_id]
        if isinstance(h5obj, h5py.Group):  # h5py.File is a subclass of h5py.Group
            self._getHardLinkIds(h5obj)
            obj_json = self._getGroup(h5obj, include_links=include_links)
        elif isinstance(h5obj, h5py.Dataset):
            obj_json = self._getDataset(h5obj)
        elif isinstance(h5obj, h5py.Datatype):
            obj_json = self._getDatatype(h5obj)
        else:
            msg = f"unexpected object type: {type(h5obj)}"
            self.log.error(msg)
            raise TypeError(msg)

        if include_attrs:
            attributes = self.getAttributes(obj_id)
            obj_json["attributes"] = attributes

        return obj_json

    def _readOpaqueDataset(self, dset):
        """ Read the full opaque dataset via the low-level API using a
        memory type tagged to match the file type.  Real HDF5 opaque data
        usually carries a "tag" (an arbitrary description string); h5py's
        high-level indexing requires the memory type's tag to match, and
        reading via a plain untagged buffer raises "no appropriate function
        for conversion path". """
        file_tid = dset.id.get_type()
        itemsize = file_tid.get_size()
        mem_tid = h5t.create(h5t.OPAQUE, itemsize)
        tag = file_tid.get_tag()
        if tag:
            mem_tid.set_tag(tag)
        buf = np.zeros(dset.shape, dtype=f"V{itemsize}")
        space = dset.id.get_space()
        dset.id.read(space, space, buf, mtype=mem_tid)
        return buf

    def getDatasetValues(self, dset_id, sel, dtype=None, query=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """

        dset = self._id_map[dset_id]
        self.log.info(f"getDatasetValues: {dset_id}")
        if dset.shape is None:
            # TBD: return something like h5py.Empty in this case?
            return None

        if query is not None:
            # h5py doesn't support query
            raise NotImplementedError("queryDataset not implemented for H5pyPlugin")

        if isOpaqueDtype(dset.dtype):
            # read the whole (tag-matched) dataset, then apply the selection
            # with plain numpy indexing - which, unlike h5py's own dataspace
            # selection, has no trouble with a paired-coordinate (multiple
            # list dims) selection.
            arr = self._readOpaqueDataset(dset)
            if sel is not None and sel.select_type != selections.H5S_SEL_ALL:
                if not isinstance(sel, selections.SimpleSelection):
                    raise NotImplementedError("selection type not supported")
                arr = arr[sel.slices]
        elif sel is None or sel.select_type == selections.H5S_SEL_ALL:
            arr = dset[...]
        elif isinstance(sel, selections.SimpleSelection):
            rank = len(sel.shape)
            slices = sel.slices
            list_dims = [d for d in range(rank) if isinstance(slices[d], list)]
            if len(list_dims) > 1:
                # h5py only supports one coordinate array at a time.
                # Decompose into n separate reads (one per paired-coordinate index)
                # then stack the results.
                list_dims_set = set(list_dims)
                n = len(slices[list_dims[0]])
                reads = []
                for i in range(n):
                    idx = tuple(
                        int(slices[d][i]) if d in list_dims_set else slices[d]
                        for d in range(rank)
                    )
                    reads.append(dset[idx])
                arr = np.stack(reads)
            else:
                arr = dset[slices]
        else:
            raise NotImplementedError("selection type not supported")

        # convert any h5py references to h5json references
        arr = self._copy_array_in(arr, fin=dset.file)
        return arr

    def getObjIdByAddress(self, addr):
        if addr in self._addr_map:
            return self._addr_map[addr]
        else:
            return None

    def get_root_id(self):
        """ Return root id """
        return self._root_id

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def flush(self):
        """ Write dirty items """
        if self.closed:
            # no db set yet
            self.log.warning("h5py_plugin - flush called but no db")
            return False
        if not self._f:
            self.log.warning("h5py_plugin file not open")
            raise IOError("open not called")
        if self._read_only:
            if self.db.new_objects or self.db.dirty_objects:
                # a read_only plugin must never write to storage, but in-memory-only
                # edits made against it (e.g. transient annotations the caller never
                # intends to persist) are fine to just leave un-flushed
                self.log.warning("read_only plugin: not persisting pending in-memory changes")
                return False
            return True  # nothing to persist, and never anything to initialize

        self.log.info("h5py_plugin.flush()")

        root_id = self.db.root_id

        if self.db.new_objects or self.db.dirty_objects or self.db.resized_datasets or self._init:
            # walk the tree whenever there's anything pending, not just brand
            # new objects - _id_map only holds objects created or discovered
            # this session, so a dirty/resized *pre-existing* object (e.g.
            # after a close()+reopen() in append mode) needs this walk to
            # reconnect it into _id_map before the update loop below can find it
            root_json = self.db.getObjectById(root_id)

            if "links" in root_json:
                root_links = root_json["links"]
                self._createObjects(self._f, root_links, visited=set((root_id,)))

        # update attributes, dataset values - iterate every object this
        # plugin has ever seen (whether created this session or discovered
        # by reading), so dirty changes to pre-existing objects are applied
        for obj_id, obj in list(self._id_map.items()):
            if self.db.is_dirty(obj_id) or self._init:
                self.updateAttributes(obj_id, obj)
                collection = getCollectionForId(obj_id)
                if collection == "datasets":
                    if self.db.is_resized(obj_id):
                        self.resizeDataset(obj_id, obj)
                    if not self.no_data:
                        # Every explicit dataset write goes through
                        # setDatasetValues() and is tracked in
                        # _getDatasetUpdates() - updateDatasetValues() applies
                        # whatever's pending directly (a no-op if nothing is)
                        # and needs no special "_init" case: a dataset with no
                        # pending update was never explicitly written, and
                        # already has correct fill-value/zero semantics from
                        # h5py's own create_dataset(fillvalue=...), with no
                        # need to redundantly re-establish it via a synthetic
                        # write (initializeDatasetValues() used to do exactly
                        # that, by re-fetching a merged view via
                        # Hdf5db.getDatasetValues() - a needlessly complex
                        # path that's actively wrong for some dtypes, e.g. a
                        # compound type with a vlen-string array field).
                        self.updateDatasetValues(obj_id, obj)
        # mark time write is complete
        # updates before this time will not need to be written
        # TBD: possible race condition with multithreading
        self._flush_time = time.time()

        self._init = False  # done with init after first flush
        return True  # all objects written successfully

    def open(self):
        """ open HDF5 file """
        self.log.debug("h5py_plugin open")
        if self.db is None:
            # no db set yet
            self.log.warning("no self.db db_ref")
            raise ValueError("no db")
        if self._f:
            return self._root_id  # already open

        if self._read_only:
            mode = 'r'
        elif self._append:
            mode = 'a'
        else:
            mode = 'w'
        self.log.info(f"opening h5py file: {self._filepath} mode: {mode}")
        self._f = h5py.File(self._filepath, mode=mode)
        if not self._read_only:
            self._append = True  # switch to append mode for next open

        if self.db.root_id:
            self._root_id = self.db.root_id
        else:
            self._root_id = createObjId(obj_type="groups")

        self._id_map[self._root_id] = self._f
        addr = h5py.h5o.get_info(self._f.id).addr
        self._addr_map[addr] = self._root_id

        return self._root_id

    def close(self):
        """ close storage handle.

        Doesn't flush - Hdf5db.close() (the only caller) always calls
        Hdf5db.flush() immediately beforehand, which itself calls this
        plugin's flush(); re-flushing here would be redundant (and, for a
        stdout-destined H5JsonPlugin, would print the dump a second time). """
        self.log.debug("h5py_plugin.close()")
        if not self._f:
            # not open
            return
        self._f.close()
        self._f = None
        self._id_map = {}
        self._addr_map = {}

    def isClosed(self):
        """ return closed status """
        return False if self._f else True

    def getStats(self):
        """ return a dictionary object with at minimum the following keys:
            'created': creation time
            'lastModified': modificationTime
            'owner': owner name
        """
        stat_info = os_stat(self.filepath)
        stats = {}
        stats['created'] = stat_info.st_ctime
        stats["lastModified"] = stat_info.st_mtime
        stats['owner'] = stat_info.st_uid  # TBD: convert to username?
        return stats

    def getFilters(self, compressors_only=False):
        """ return list of filters supported by h5py  """

        h5py_filters = ["H5Z_FILTER_DEFLATE", ]

        if not compressors_only:
            h5py_filters.append("H5Z_FILTER_SHUFFLE")
            h5py_filters.append("H5Z_FILTER_FLETCHER32")
            h5py_filters.append("H5Z_FILTER_SZIP")
            h5py_filters.append("H5Z_FILTER_NBIT")
            h5py_filters.append("H5Z_FILTER_SCALEOFFSET")

        return tuple(h5py_filters)
