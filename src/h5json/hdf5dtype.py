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

import struct

import numpy as np

from .objid import getHashTagForId, getUuidFromId
from .selections import Selection, from_region_json, from_dict


# --- RegionReference.tobytes()/frombytes() binary format ---
#
# magic(4) + version(1) + id_len(2) + id_bytes + sel_len(4) + sel_bytes + trailer(1)
#
# The trailer is a fixed non-zero byte.  It guards against a RegionReference
# ever being embedded in a numpy fixed-length byte-string ("S<n>") dtype
# value: numpy silently strips *trailing* NUL bytes from "S" dtype values on
# read, and our own struct-packed fields can legitimately end in NUL (e.g. a
# zero length or a small integer).  Since only a *trailing* run of NULs is
# stripped, ending the blob on a guaranteed non-zero byte protects everything
# before it, including internal NULs and NUL padding from an oversized dtype.
# special_dtype(ref=RegionReference) itself uses a variable-length ("O")
# dtype - see below - so this only matters if the serialized bytes are
# stored somewhere else that uses fixed-width byte strings (e.g. eventually
# writing an actual HDF5 file).
_RREF_MAGIC = b"HRRF"
_RREF_VERSION = 1
_RREF_TRAILER = b"\xff"


numpy_integer_types = (np.int8, np.uint8, np.int16, np.int16, np.int32, np.uint32, np.int64, np.uint64)
numpy_float_types = (np.float16, np.float32, np.float64)


class Reference:
    """
    Represents an HDF5 object reference
    """

    @property
    def id(self):
        """Low-level identifier appropriate for this object"""
        return self._id

    def __init__(self, bind):
        """Create a new reference by binding to
        a uuid
        """
        if not bind:
            self._id = None
        else:
            if isinstance(bind, bytes):
                bind = bind.decode()

            if not isinstance(bind, str):
                raise TypeError("Expected string id")

            if bind.find('/') != -1:
                parts = bind.split('/')
                if parts[0] not in ("groups", "datasets", "datatypes"):
                    raise TypeError("Expected id to start with 'groups/', 'datasets/' or 'datatypes/'")
                # NOTE: keep the "<collection>/" prefix intact - getHashTagForId()
                # needs it to determine the right prefix character for a bare
                # (schema 1 style) uuid; it already handles stripping the "/"
                # itself, and an already-prefixed hashtag id passes through as-is.
            self._id = getHashTagForId(bind)

    def __repr__(self):
        # return canonical uuid
        return f"{self._id}"

    def tolist(self):
        if type(self._id) is not str:
            raise TypeError("Expected string id")
        if not self._id:
            return [("",),]

        objtype_code = self._id[0]
        if objtype_code == "d":
            return [
                ("datasets/" + self._id),
            ]
        elif objtype_code == "g":
            return [
                ("groups/" + self._id),
            ]
        elif objtype_code == "t":
            return [
                ("datatypes/" + self._id),
            ]
        else:
            raise TypeError("Unexpected id type")


class RegionReference:
    """
    Represents an HDF5 region reference: the id of the referenced dataset
    plus a (binary-serialized) selections.Selection on that dataset.
    """

    @property
    def id(self):
        """Low-level identifier of the referenced dataset"""
        return self._id

    @property
    def selection_bytes(self):
        """Serialized selection (selections.Selection.tobytes()), or None if unbound"""
        return self._selection_bytes

    def __init__(self, bind=None, selection=None):
        """Create a new region reference, optionally binding immediately -
        see bind() for the meaning of the arguments.
        """
        self._id = None
        self._selection_bytes = None
        if bind is not None or selection is not None:
            self.bind(bind, selection)

    def bind(self, objid, selection=None):
        """Bind this region reference to a dataset id and a selection on it.

        objid
            The id of the referenced dataset: a uuid string/bytes (optionally
            prefixed with "datasets/", as with Reference), or an object
            exposing an `_id` attribute (e.g. a dataset object).

        selection
            A selections.Selection instance, or bytes/bytearray already
            produced by Selection.tobytes().  May be None, in which case
            the reference is left without a selection (whole dataset).
        """
        if hasattr(objid, "_id"):
            objid = objid._id

        if not objid:
            self._id = None
        else:
            if isinstance(objid, bytes):
                objid = objid.decode()
            if not isinstance(objid, str):
                raise TypeError("Expected string id")

            if objid.find("/") != -1:
                parts = objid.split("/")
                if parts[0] != "datasets":
                    raise TypeError("Expected id to start with 'datasets/'")
                # NOTE: keep the "datasets/" prefix intact - getHashTagForId()
                # needs it to determine the right prefix character for a bare
                # (schema 1 style) uuid; it already handles stripping the "/"
                # itself, and an already-prefixed hashtag id passes through as-is.
            self._id = getHashTagForId(objid)

        if selection is None:
            self._selection_bytes = None
        elif isinstance(selection, (bytes, bytearray)):
            self._selection_bytes = bytes(selection)
        elif isinstance(selection, Selection):
            self._selection_bytes = bytes(selection.tobytes())
        else:
            raise TypeError("Expected a Selection instance or serialized bytes")

        return self

    def tobytes(self):
        """ Serialize this region reference (dataset id + selection bytes) to a
        flat bytes blob, suitable for storage as a raw H5T_REFERENCE dataset
        or attribute value. """
        id_bytes = self._id.encode("ascii") if self._id else b""
        sel_bytes = self._selection_bytes if self._selection_bytes else b""
        buf = bytearray()
        buf += _RREF_MAGIC
        buf += struct.pack("<B", _RREF_VERSION)
        buf += struct.pack("<H", len(id_bytes))
        buf += id_bytes
        buf += struct.pack("<I", len(sel_bytes))
        buf += sel_bytes
        buf += _RREF_TRAILER
        return bytes(buf)

    @classmethod
    def frombytes(cls, data):
        """ Reconstruct a RegionReference from a bytes blob produced by tobytes(). """
        data = bytes(data)
        if data[:4] != _RREF_MAGIC:
            raise ValueError("Invalid region reference byte stream")
        version = struct.unpack_from("<B", data, 4)[0]
        if version != _RREF_VERSION:
            raise ValueError(f"Unsupported region reference serialization version: {version}")

        offset = 5
        id_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        id_str = data[offset:offset + id_len].decode("ascii")
        offset += id_len

        sel_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        sel_bytes = data[offset:offset + sel_len]

        ref = cls()
        ref._id = id_str if id_str else None
        ref._selection_bytes = sel_bytes if sel_bytes else None
        return ref

    def to_json(self):
        """ Convert this region reference to the h5json JSON representation:
        {"id": <uuid>, "select_type": ..., "selection": [...]} - see
        data/json/regionref_dset.json for an example.  If no selection is
        bound (see bind()) - e.g. a region reference read from an actual
        HDF5 file, where only the target dataset's identity can be
        recovered, not its selection - just {"id": <uuid>} is returned.

        Real HDF5 region references only ever select points or hyperslabs
        (possibly several disjoint blocks), so that's the representation
        used whenever possible.  H5S_SEL_FANCY (a mixed slice/coordinate
        selection) and stepped hyperslabs have no equivalent there - it's
        purely an artifact of this project's own Selection model, not a
        concept HDF5 dataspaces have - so for those, the fully general
        Selection.to_dict() representation is embedded instead, under a
        "selection_dict" key.
        """
        if self._id is None:
            raise ValueError("Cannot convert a null region reference to JSON")
        d = {"id": getUuidFromId(self._id)}
        if self._selection_bytes is None:
            return d
        sel = Selection.frombytes(self._selection_bytes)
        try:
            d.update(sel.to_region_json())
        except NotImplementedError:
            d["selection_dict"] = sel.to_dict()
        return d

    @classmethod
    def from_json(cls, d):
        """ Reconstruct a RegionReference from the h5json JSON representation
        produced by to_json() - {"id": <uuid>, "select_type": ..., "selection": [...]},
        {"id": <uuid>, "selection_dict": {...}}, or just {"id": <uuid>} for a
        reference with no selection - or None for a null reference.
        """
        if d is None:
            return cls()
        if "id" not in d:
            raise KeyError("expected 'id' key in region reference JSON")
        if "selection_dict" in d:
            sel = from_dict(d["selection_dict"])
        elif "select_type" in d:
            sel = from_region_json(d)
        else:
            sel = None
        ref = cls()
        ref.bind("datasets/" + d["id"], sel)
        return ref

    def __repr__(self):
        return "<HDF5 region reference>"


def is_reference(val):
    """ Return True if the type or value is a Reference """

    if isinstance(val, object) and val.__class__.__name__ == "Reference":
        return True
    elif isinstance(val, type) and val.__name__ == "Reference":
        return True
    else:
        return False


def is_regionreference(val):
    """ Return True if the type or value is a RegionReference """

    if isinstance(val, object) and val.__class__.__name__ == "RegionReference":
        return True
    elif isinstance(val, type) and val.__name__ == "RegionReference":
        return True

    return False


def has_reference(dtype):
    """ return True if the dtype (or a sub-type) is a Reference or RegionReference type """
    has_ref = False
    if not isinstance(dtype, np.dtype):
        return False
    if len(dtype) > 0:
        for name in dtype.fields:
            item = dtype.fields[name]
            if has_reference(item[0]):
                has_ref = True
                break
    elif dtype.metadata and "ref" in dtype.metadata:
        basedt = dtype.metadata["ref"]
        has_ref = is_reference(basedt) or is_regionreference(basedt)
    elif dtype.metadata and "vlen" in dtype.metadata:
        basedt = dtype.metadata["vlen"]
        has_ref = has_reference(basedt)
    return has_ref


def special_dtype(**kwds):
    """Create a new h5py "special" type.  Only one keyword may be given.

    Legal keywords are:

    vlen = basetype
        Base type for HDF5 variable-length datatype. This can be Python
        str type or instance of np.dtype.
        Example: special_dtype( vlen=str )

    enum = (basetype, values_dict)
        Create a NumPy representation of an HDF5 enumerated type.  Provide
        a 2-tuple containing an (integer) base dtype and a dict mapping
        string names to integer values.

    ref = Reference | RegionReference
        Create a NumPy representation of an HDF5 object or region reference
        type.  Reference is a fixed-size ("S48") type, since it only ever
        holds an object id.  RegionReference is a variable-length ("O")
        type, since its size depends on the bound selection, not just the
        referenced dataset - see RegionReference.tobytes()."""

    if len(kwds) != 1:
        raise TypeError("Exactly one keyword may be provided")

    name, val = kwds.popitem()

    if name == "vlen":

        return np.dtype("O", metadata={"vlen": val})

    if name == "enum":

        try:
            dt, enum_vals = val
        except TypeError:
            msg = "Enums must be created from a 2-tuple "
            msg += "(basetype, values_dict)"
            raise TypeError(msg)

        dt = np.dtype(dt)
        if dt.kind not in "iu":
            raise TypeError("Only integer types can be used as enums")

        return np.dtype(dt, metadata={"enum": enum_vals})

    if name == "ref":
        dt = None
        if val is Reference:
            dt = np.dtype("S48", metadata={"ref": Reference})
        elif val is RegionReference:
            dt = np.dtype("O", metadata={"ref": RegionReference})
        else:
            raise ValueError("Ref class must be Reference or RegionReference")

        return dt

    raise TypeError(f'Unknown special type "{name}"')


def find_item_type(data):
    """Find the item type of a simple object or collection of objects.

    E.g. [[['a']]] -> str

    The focus is on collections where all items have the same type; we'll return
    None if that's not the case.

    The aim is to treat numpy arrays of Python objects like normal Python
    collections, while treating arrays with specific dtypes differently.
    We're also only interested in array-like collections - lists and tuples,
    possibly nested - not things like sets or dicts.
    """
    if isinstance(data, np.ndarray):
        if (
            data.dtype.kind == 'O' and not check_dtype(vlen=data.dtype)
        ):
            item_types = {type(e) for e in data.flat}
        else:
            return None
    elif isinstance(data, (list, tuple)):
        item_types = {find_item_type(e) for e in data}
    else:
        return type(data)

    if len(item_types) != 1:
        return None
    return item_types.pop()


def guess_dtype(data):
    """ Attempt to guess an appropriate dtype for the object, returning None
    if nothing is appropriate (or if it should be left up the the array
    constructor to figure out)
    """

    # todo - handle RegionReference, Reference
    item_type = find_item_type(data)
    if item_type is bytes:
        return special_dtype(vlen=bytes)
    if item_type is str:
        return special_dtype(vlen=str)

    return None


def is_float16_dtype(dt):
    if dt is None:
        return False

    dt = np.dtype(dt)  # normalize strings -> np.dtype objects
    return dt.kind == 'f' and dt.itemsize == 2


def check_dtype(**kwds):
    """Check a dtype for h5py special type "hint" information.  Only one
    keyword may be given.

    vlen = dtype
        If the dtype represents an HDF5 vlen, returns the Python base class.
        Returns None if the dtype does not represent an HDF5 vlen.

    enum = dtype
        If the dtype represents an HDF5 enumerated type, returns the dictionary
        mapping string names to integer values.  Returns None if the dtype does
        not represent an HDF5 enumerated type.

    ref = dtype
        If the dtype represents an HDF5 reference type, returns the reference
        class (either Reference or RegionReference).  Returns None if the dtype
        does not represent an HDF5 reference type.
    """

    if len(kwds) != 1:
        raise TypeError("Exactly one keyword may be provided")

    name, dt = kwds.popitem()

    if name not in ("vlen", "enum", "ref"):
        raise TypeError(f"Unknown special type {name}")

    try:
        return dt.metadata[name]
    except TypeError:
        return None
    except KeyError:
        return None


def getTypeResponse(typeItem):
    """
    Convert the given type item  to a predefined type string for
        predefined integer and floating point types ("H5T_STD_I64LE", et. al).
        For compound types, recursively iterate through the typeItem and do
        same conversion for fields of the compound type."""
    response = None
    if "uuid" in typeItem:
        # committed type, just return uuid
        response = "datatypes/" + typeItem["uuid"]
    elif typeItem["class"] in ("H5T_INTEGER", "H5T_FLOAT"):
        # just return the class and base for pre-defined types
        response = {}
        response["class"] = typeItem["class"]
        response["base"] = typeItem["base"]
    elif typeItem["class"] == "H5T_OPAQUE":
        response = {}
        response["class"] = "H5T_OPAQUE"
        response["size"] = typeItem["size"]
    elif typeItem["class"] == "H5T_REFERENCE":
        response = {}
        response["class"] = "H5T_REFERENCE"
        response["base"] = typeItem["base"]
    elif typeItem["class"] == "H5T_COMPOUND":
        response = {}
        response["class"] = "H5T_COMPOUND"
        fieldList = []
        for field in typeItem["fields"]:
            fieldItem = {}
            fieldItem["name"] = field["name"]
            fieldItem["type"] = getTypeResponse(field["type"])  # recurse call
            fieldList.append(fieldItem)
        response["fields"] = fieldList
    else:
        response = {}  # otherwise, return full type
        for k in typeItem.keys():
            if k == "base":
                if isinstance(typeItem[k], dict):
                    response[k] = getTypeResponse(typeItem[k])  # recursive call
                else:
                    response[k] = typeItem[k]  # predefined type
            elif k not in ("size", "base_size"):
                response[k] = typeItem[k]
    return response


def getTypeItem(dt, metadata=None):
    """
    Return type info.
          For primitive types, return string with typename
          For compound types return array of dictionary items
    """
    predefined_int_types = {
        "int8": "H5T_STD_I8",
        "uint8": "H5T_STD_U8",
        "int16": "H5T_STD_I16",
        "uint16": "H5T_STD_U16",
        "int32": "H5T_STD_I32",
        "uint32": "H5T_STD_U32",
        "int64": "H5T_STD_I64",
        "uint64": "H5T_STD_U64",
    }
    predefined_float_types = {
        "float16": "H5T_IEEE_F16",
        "float32": "H5T_IEEE_F32",
        "float64": "H5T_IEEE_F64",
    }

    dt = np.dtype(dt)  # convert 'int32', np.int32, etc. to a dtype

    if not metadata and dt.metadata:
        metadata = dt.metadata

    type_info = {}
    if len(dt):
        # compound type
        names = dt.names
        type_info["class"] = "H5T_COMPOUND"
        fields = []
        for name in names:
            field = {"name": name}
            field["type"] = getTypeItem(dt[name])
            fields.append(field)
            type_info["fields"] = fields
    elif dt.shape:
        # array type
        if dt.base == dt:
            raise TypeError("Expected base type to be different than parent")
        # array type
        type_info["dims"] = dt.shape
        type_info["class"] = "H5T_ARRAY"
        type_info["base"] = getTypeItem(dt.base, metadata=metadata)
    elif dt.kind == "O":
        # vlen string or data
        #
        # check for h5py variable length extension
        vlen_check = vlenBaseType(dt)

        if metadata and "ref" in metadata:
            ref_check = metadata["ref"]
        else:
            ref_check = check_dtype(ref=dt.base)

        if vlen_check == bytes:
            type_info["class"] = "H5T_STRING"
            type_info["length"] = "H5T_VARIABLE"
            type_info["charSet"] = "H5T_CSET_ASCII"
            type_info["strPad"] = "H5T_STR_NULLTERM"
        elif vlen_check == str:
            type_info["class"] = "H5T_STRING"
            type_info["length"] = "H5T_VARIABLE"
            type_info["charSet"] = "H5T_CSET_UTF8"
            type_info["strPad"] = "H5T_STR_NULLTERM"
        elif isinstance(vlen_check, np.dtype):
            # vlen data
            type_info["class"] = "H5T_VLEN"
            type_info["size"] = "H5T_VARIABLE"
            type_info["base"] = getTypeItem(vlen_check)
        elif vlen_check is not None:
            #  unknown vlen type
            raise TypeError("Unknown h5py vlen type: " + str(vlen_check))
        elif ref_check is not None:
            # a reference type
            type_info["class"] = "H5T_REFERENCE"

            if ref_check.__name__ == "Reference":
                type_info["base"] = "H5T_STD_REF_OBJ"  # objref
            elif ref_check.__name__ == "RegionReference":
                type_info["base"] = "H5T_STD_REF_DSETREG"  # region ref
            else:
                raise TypeError("unexpected reference type")
        else:
            raise TypeError("unknown object type")
    elif dt.kind == "T":
        # numpy StringDType - accepted as a convenience input, but reported
        # identically to the "O"-kind vlen str case above: h5json's type
        # descriptor has no way to distinguish the two (and neither does
        # HDF5 itself), so there's nothing to preserve by treating it
        # differently
        type_info["class"] = "H5T_STRING"
        type_info["length"] = "H5T_VARIABLE"
        type_info["charSet"] = "H5T_CSET_UTF8"
        type_info["strPad"] = "H5T_STR_NULLTERM"
    elif dt.kind == "V":
        # void type - the only state h5json tracks for opaque data is its
        # size; "tag" is an optional, arbitrary HDF5 description string with
        # no numpy-dtype equivalent, so it's omitted rather than emitted
        # empty (an empty "tag" violates the schema's minLength: 1).
        type_info["class"] = "H5T_OPAQUE"
        type_info["size"] = dt.itemsize
    elif dt.base.kind == "S":
        # check for object reference
        ref_check = check_dtype(ref=dt.base)
        if ref_check is not None:
            # a reference type
            type_info["class"] = "H5T_REFERENCE"

            if ref_check is Reference:
                type_info["base"] = "H5T_STD_REF_OBJ"  # objref
                type_info["length"] = dt.itemsize
            elif ref_check is RegionReference:
                type_info["base"] = "H5T_STD_REF_DSETREG"  # region ref
                # unlike an object ref, a region ref's size depends on the
                # bound selection, not just the referenced dataset - it can't
                # be reported as a fixed value (mirrors vlen strings/types)
                type_info["length"] = "H5T_VARIABLE"
            else:
                raise TypeError("unexpected reference type")
        else:
            # Fixed length string type
            type_info["class"] = "H5T_STRING"
            type_info["length"] = dt.itemsize
        if ref_check is None and metadata and metadata.get("h5py_encoding") == "utf-8":
            # h5py tags a fixed-length string dtype's desired charset via
            # this metadata key (see string_dtype()) - numpy's 'S' dtype
            # itself has no notion of charset, so this is the only place
            # that information survives to be read back here.
            type_info["charSet"] = "H5T_CSET_UTF8"
        else:
            type_info["charSet"] = "H5T_CSET_ASCII"
        type_info["strPad"] = "H5T_STR_NULLPAD"
    elif dt.base.kind == "U":
        # Fixed length unicode type
        ref_check = check_dtype(ref=dt.base)
        if ref_check is not None:
            raise TypeError("unexpected reference type")

        # Fixed length string type with unicode support
        type_info["class"] = "H5T_STRING"

        # this can be problematic if the encoding of the string is not valid,
        # or reqires too many bytes.  Use variable length strings to handle all
        # UTF8 strings correctly
        type_info["charSet"] = "H5T_CSET_UTF8"
        # convert from UTF32 length to a fixed length
        type_info["length"] = dt.itemsize
        type_info["strPad"] = "H5T_STR_NULLPAD"

    elif dt.kind == "b":
        # boolean type - h5py stores as enum
        # assume LE unless the numpy byteorder is '>'
        byteorder = "LE"
        if dt.base.byteorder == ">":
            byteorder = "BE"
        # this mapping is an h5py convention for boolean support
        bool_false = {"name": "FALSE", "value": 0}
        bool_true = {"name": "TRUE", "value": 1}
        members = [bool_false, bool_true]
        type_info["class"] = "H5T_ENUM"
        type_info["members"] = members
        base_info = {"class": "H5T_INTEGER"}
        base_info["base"] = "H5T_STD_I8" + byteorder
        type_info["base"] = base_info
    elif dt.kind == "f":
        # floating point type
        type_info["class"] = "H5T_FLOAT"
        byteorder = "LE"
        if dt.byteorder == ">":
            byteorder = "BE"
        if dt.name in predefined_float_types:
            # maps to one of the HDF5 predefined types
            float_type = predefined_float_types[dt.base.name]
            type_info["base"] = float_type + byteorder
        else:
            raise TypeError("Unexpected floating point type: " + dt.name)
    elif dt.kind == "i" or dt.kind == "u":
        # integer type

        # assume LE unless the numpy byteorder is '>'
        byteorder = "LE"
        if dt.base.byteorder == ">":
            byteorder = "BE"

        # numpy integer type - but check to see if this is the hypy
        # enum extension
        if metadata and "enum" in metadata:
            # yes, this is an enum!
            mapping = metadata["enum"]
            type_info["class"] = "H5T_ENUM"
            members = []
            for name in mapping:
                value = mapping[name]
                item = {"name": name, "value": value}
                members.append(item)
            type_info["members"] = members
            if dt.name not in predefined_int_types:
                raise TypeError("Unexpected integer type: " + dt.name)
            # maps to one of the HDF5 predefined types
            base_info = {"class": "H5T_INTEGER"}
            base_info["base"] = predefined_int_types[dt.name] + byteorder
            type_info["base"] = base_info
        else:
            type_info["class"] = "H5T_INTEGER"
            base_name = dt.name

            if dt.name not in predefined_int_types:
                raise TypeError("Unexpected integer type: " + dt.name)

            type_info["base"] = predefined_int_types[base_name] + byteorder

    else:
        # unexpected kind
        raise TypeError(f"unexpected dtype kind: {dt.kind}")

    return type_info


def isVlen(dt):
    """
    Return True if the type contains variable length elements
    """
    is_vlen = False
    if len(dt):
        names = dt.names
        for name in names:
            if isVlen(dt[name]):
                is_vlen = True
                break
    else:
        if dt.base.metadata and "vlen" in dt.base.metadata:
            is_vlen = True
    return is_vlen


def vlenBaseType(dt):
    """
    Return the base dtype of a vlen, otherwise none
    """
    if len(dt):
        raise TypeError("BaseType can't be deterined for compound type")
    if dt.base.metadata and "vlen" in dt.base.metadata:
        base_dt = dt.base.metadata["vlen"]
        if base_dt not in (bytes, str):
            base_dt = np.dtype(base_dt)
    else:
        base_dt = None
    return base_dt


def isOpaqueDtype(dt):
    """
    Return True if this is an opaque dtype
    """
    if dt.kind == "V" and len(dt) == 0 and len(dt.shape) == 0 and not dt.names:
        return True
    if dt.metadata and dt.metadata.get('h5py_opaque'):
        return True
    return False


def getItemSize(typeItem):
    """
    Get size of an item in bytes.
        For variable length types (e.g. variable length strings),
        return the string "H5T_VARIABLE"
    """
    # handle the case where we are passed a primitive type first
    if isinstance(typeItem, str) or isinstance(typeItem, bytes):
        for type_prefix in ("H5T_STD_I", "H5T_STD_U", "H5T_IEEE_F"):
            if typeItem.startswith(type_prefix):
                nlen = len(type_prefix)
                num_bits = typeItem[nlen:]
                if num_bits[-2:] in ("LE", "BE"):
                    num_bits = num_bits[:-2]
                try:
                    return int(num_bits) // 8
                except ValueError:
                    raise TypeError("Invalid Type")
        # none of the expect primative types mathched
        raise TypeError("Invalid Type")
    if not isinstance(typeItem, dict):
        raise TypeError("invalid type")

    item_size = 0
    if "class" not in typeItem:
        raise KeyError("'class' not provided")
    typeClass = typeItem["class"]

    if typeClass == "H5T_INTEGER":
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        item_size = getItemSize(typeItem["base"])

    elif typeClass == "H5T_FLOAT":
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        item_size = getItemSize(typeItem["base"])

    elif typeClass == "H5T_STRING":
        if "length" not in typeItem:
            raise KeyError("'length' not provided")
        item_size = typeItem["length"]

    elif typeClass == "H5T_VLEN":
        item_size = "H5T_VARIABLE"
    elif typeClass == "H5T_OPAQUE":
        if "size" not in typeItem:
            raise KeyError("'size' not provided")
        item_size = int(typeItem["size"])

    elif typeClass == "H5T_ARRAY":
        if "dims" not in typeItem:
            raise KeyError("'dims' must be provided for array types")
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        item_size = getItemSize(typeItem["base"])

    elif typeClass == "H5T_ENUM":
        if "base" not in typeItem:
            raise KeyError("'base' must be provided for enum types")
        item_size = getItemSize(typeItem["base"])

    elif typeClass == "H5T_REFERENCE":
        if typeItem.get("base") == "H5T_STD_REF_DSETREG":
            # a region ref's size depends on the bound selection, not just
            # the referenced dataset, so it can't be reported as a fixed
            # value - same convention as vlen strings/types
            item_size = "H5T_VARIABLE"
        elif "length" in typeItem:
            item_size = typeItem["length"]
        elif typeItem.get("base") == "H5T_STD_REF_OBJ":
            # obj ref values are in the form: "groups/<id>" or
            # "datasets/<id>" or "datatypes/<id>"
            item_size = 48
        else:
            item_size = 80  # tb: just take a guess at this for now
    elif typeClass == "H5T_COMPOUND":
        if "fields" not in typeItem:
            raise KeyError("'fields' not provided for compound type")
        fields = typeItem["fields"]
        if not isinstance(fields, list):
            raise TypeError("Type Error: expected list type for 'fields'")
        if not fields:
            raise KeyError("no 'field' elements provided")
        # add up the size of each sub-field
        for field in fields:
            if not isinstance(field, dict):
                raise TypeError("Expected dictionary type for field")
            if "type" not in field:
                raise KeyError("'type' missing from field")
            subtype_size = getItemSize(field["type"])  # recursive call
            if subtype_size == "H5T_VARIABLE":
                item_size = "H5T_VARIABLE"
                break  # don't need to look at the rest

            item_size += subtype_size
    else:
        raise TypeError("Invalid type class")

    # calculate array type
    if "dims" in typeItem and isinstance(item_size, int):
        dims = typeItem["dims"]
        for dim in dims:
            item_size *= dim

    return item_size


def getDtypeItemSize(dtype):
    """ Return size of dtype in bytes
        For variable length types (e.g. variable length strings),
        return the string "H5T_VARIABLE
    """
    item_size = 0
    if len(dtype):
        # compound dtype
        for i in range(len(dtype)):
            sub_dt = dtype[i]
            sub_dt_size = getDtypeItemSize(sub_dt)
            if sub_dt_size == "H5T_VARIABLE":
                item_size = "H5T_VARIABLE"  # return variable if any component is variable
                break
            item_size += sub_dt_size
    else:
        # primitive type
        if dtype.shape:
            base_size = getDtypeItemSize(dtype.base)
            if base_size == "H5T_VARIABLE":
                item_size = "H5T_VARIABLE"
            else:
                nelements = np.prod(dtype.shape)
                item_size = base_size * nelements
        else:
            if dtype.metadata and (
                "vlen" in dtype.metadata or dtype.metadata.get("ref") is RegionReference
            ):
                # RegionReference is stored as a length-prefixed opaque byte
                # blob (see array_util._isVlenLike()/RegionReference.tobytes()),
                # unlike a plain object Reference (a fixed-format id string).
                item_size = "H5T_VARIABLE"
            else:
                item_size = dtype.itemsize
    return item_size


def getNumpyTypename(hdf5TypeName, typeClass=None):
    predefined_int_types = {
        "H5T_STD_I8": "i1",
        "H5T_STD_U8": "u1",
        "H5T_STD_I16": "i2",
        "H5T_STD_U16": "u2",
        "H5T_STD_I32": "i4",
        "H5T_STD_U32": "u4",
        "H5T_STD_I64": "i8",
        "H5T_STD_U64": "u8",
    }
    predefined_float_types = {
        "H5T_IEEE_F16": "f2",
        "H5T_IEEE_F32": "f4",
        "H5T_IEEE_F64": "f8",
    }

    if len(hdf5TypeName) < 3:
        raise Exception("Type Error: invalid typename: ")
    endian = "<"  # default endian
    key = hdf5TypeName
    if hdf5TypeName.endswith("LE"):
        key = hdf5TypeName[:-2]
    elif hdf5TypeName.endswith("BE"):
        key = hdf5TypeName[:-2]
        endian = ">"

    if key in predefined_int_types and (
        typeClass is None or typeClass == "H5T_INTEGER"
    ):
        return endian + predefined_int_types[key]
    if key in predefined_float_types and (
        typeClass is None or typeClass == "H5T_FLOAT"
    ):
        return endian + predefined_float_types[key]
    raise TypeError("Type Error: invalid type")


def createBaseDataType(typeItem):
    dtRet = None
    if isinstance(typeItem, str):
        # should be one of the predefined types
        dtName = getNumpyTypename(typeItem)
        dtRet = np.dtype(dtName)
        return dtRet  # return predefined type

    if not isinstance(typeItem, dict):
        raise TypeError("Type Error: invalid type")

    if "class" not in typeItem:
        raise KeyError("'class' not provided")
    typeClass = typeItem["class"]

    dims = ""
    if "dims" in typeItem:
        if typeClass != "H5T_ARRAY":
            raise TypeError("'dims' only supported for integer types")

        dims = None
        if isinstance(typeItem["dims"], int):
            dims = typeItem["dims"]  # make into a tuple
        elif not isinstance(typeItem["dims"], list) and not isinstance(
            typeItem["dims"], tuple
        ):
            raise TypeError("expected list or integer for dims")
        else:
            dims = typeItem["dims"]
        dims = str(tuple(dims))

    if typeClass == "H5T_INTEGER":
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        baseType = getNumpyTypename(typeItem["base"], typeClass="H5T_INTEGER")
        dtRet = np.dtype(dims + baseType)
    elif typeClass == "H5T_FLOAT":
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        baseType = getNumpyTypename(typeItem["base"], typeClass="H5T_FLOAT")
        dtRet = np.dtype(dims + baseType)
    elif typeClass == "H5T_STRING":
        if "length" not in typeItem:
            raise KeyError("'length' not provided")
        if "charSet" not in typeItem:
            raise KeyError("'charSet' not provided")

        if typeItem["length"] == "H5T_VARIABLE":
            if dims:
                msg = "ArrayType is not supported for variable len types"
                raise TypeError(msg)
            if typeItem["charSet"] == "H5T_CSET_ASCII":
                dtRet = special_dtype(vlen=bytes)
            elif typeItem["charSet"] == "H5T_CSET_UTF8":
                dtRet = special_dtype(vlen=str)
            else:
                raise TypeError("unexpected 'charSet' value")
        else:
            nStrSize = typeItem["length"]
            if not isinstance(nStrSize, int):
                raise TypeError("expecting integer value for 'length'")
            # a fixed size string - use the "S" type code regardless of
            # declared charset (otherwise numpy would reserve bytes for a
            # UTF32 representation)
            dtRet = np.dtype(dims + "S" + str(nStrSize))
            if typeItem["charSet"] == "H5T_CSET_UTF8":
                # h5py tags a fixed-length UTF8-declared string dtype via
                # this metadata key (see h5type.string_dtype()) - numpy's
                # 'S' dtype has no charset of its own, so this is the only
                # way it survives a round trip through JSON.
                dtRet = np.dtype(dtRet, metadata={"h5py_encoding": "utf-8"})
            elif typeItem["charSet"] != "H5T_CSET_ASCII":
                raise TypeError("unexpected 'charSet' value")
    elif typeClass == "H5T_VLEN":
        if dims:
            msg = "ArrayType is not supported for variable len types"
            raise TypeError(msg)
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        # base may itself be a compound (or other non-base) type, which
        # only createDataType() knows how to dispatch - mirrors the
        # H5T_ARRAY branch below.
        baseType = createDataType(typeItem["base"])
        dtRet = special_dtype(vlen=np.dtype(baseType))
    elif typeClass == "H5T_OPAQUE":
        if dims:
            msg = "Opaque Type is not supported for variable len types"
            raise TypeError(msg)
        if "size" not in typeItem:
            raise KeyError("'size' not provided")
        nSize = int(typeItem["size"])
        if nSize <= 0:
            raise TypeError("'size' must be non-negative")
        dtRet = np.dtype("V" + str(nSize))
    elif typeClass == "H5T_ARRAY":
        if not dims:
            raise KeyError("'dims' must be provided for array types")
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        arrayBaseType = typeItem["base"]
        if isinstance(arrayBaseType, dict):
            if "class" not in arrayBaseType:
                raise KeyError("'class' not provided for array base type")
            type_classes = ("H5T_INTEGER", "H5T_FLOAT", "H5T_STRING", "H5T_COMPOUND", "H5T_ARRAY")
            if arrayBaseType["class"] not in type_classes:
                msg = "Array Type base type must be integer, float, string, compound or array"
                raise TypeError(msg)
        baseType = createDataType(arrayBaseType)
        if isinstance(typeItem["dims"], int):
            dims = typeItem["dims"]  # make into a tuple
        elif type(typeItem["dims"]) not in (list, tuple):
            raise TypeError("expected list or integer for dims")
        else:
            dims = typeItem["dims"]
        # create an array type of the base type

        dtRet = np.dtype((baseType, dims))
        """
        metadata = None
        if baseType.metadata:
            metadata = dict(baseType.metadata)
            dtRet = np.dtype(dims + baseType.str, metadata=metadata)
        else:
            dtRet = np.dtype(dims + baseType.str)
        return dtRet  # return predefined type
        """
    elif typeClass == "H5T_REFERENCE":
        if "base" not in typeItem:
            raise KeyError("'base' not provided")
        if typeItem["base"] == "H5T_STD_REF_OBJ":
            dtRet = special_dtype(ref=Reference)
        elif typeItem["base"] == "H5T_STD_REF_DSETREG":
            dtRet = special_dtype(ref=RegionReference)
        else:
            raise TypeError("Invalid base type for reference type")

    elif typeClass == "H5T_ENUM":
        if "base" not in typeItem:
            raise KeyError("Expected 'base' to be provided for enum type")
        base_json = typeItem["base"]
        if "class" not in base_json:
            raise KeyError("Expected class field in base type")
        if base_json["class"] != "H5T_INTEGER":
            msg = "Only integer base types can be used with enum type"
            raise TypeError(msg)
        if "mapping" in typeItem:
            mapping = typeItem["mapping"]
        elif "members" in typeItem:
            mapping = typeItem["members"]  # backward-compatibility for hdf5-json
        else:
            raise KeyError("'mapping' not provided for enum type")

        if len(mapping) == 0:
            raise KeyError("empty enum map")

        dt = createBaseDataType(base_json)
        if isinstance(mapping, list):
            # convert to a dictionary
            values_dict = dict((m["name"], m["value"]) for m in mapping)
        elif isinstance(mapping, dict):
            # just use as is
            values_dict = mapping
        else:
            raise TypeError("Expected dict or list mapping for enum type")

        if all(
            (
                dt.kind == "i",
                dt.name == "int8",
                len(mapping) == 2,
                "TRUE" in values_dict,
                "FALSE" in values_dict,
            )
        ):
            # convert to numpy boolean type
            dtRet = np.dtype("bool")
        else:
            # not a boolean enum, use h5py special dtype
            dtRet = special_dtype(enum=(dt, values_dict))

    else:
        raise TypeError("Invalid type class")

    return dtRet


def createDataType(typeItem):
    """
    Create a numpy datatype given a json type
    """
    dtRet = None
    if type(typeItem) in (str, bytes):
        # should be one of the predefined types
        dtName = getNumpyTypename(typeItem)
        dtRet = np.dtype(dtName)
        return dtRet  # return predefined type

    if not isinstance(typeItem, dict):
        raise TypeError("invalid type")

    if "class" not in typeItem:
        raise KeyError("'class' not provided")
    typeClass = typeItem["class"]

    if typeClass == "H5T_COMPOUND":
        if "fields" not in typeItem:
            raise KeyError("'fields' not provided for compound type")
        fields = typeItem["fields"]
        if type(fields) is not list:
            raise TypeError("Type Error: expected list type for 'fields'")
        if not fields:
            raise KeyError("no 'field' elements provided")
        subtypes = []
        for field in fields:

            if not isinstance(field, dict):
                raise TypeError("Expected dictionary type for field")
            if "name" not in field:
                raise KeyError("'name' missing from field")
            if "type" not in field:
                raise KeyError("'type' missing from field")
            field_name = field["name"]
            if not isinstance(field_name, str):
                raise TypeError("field names must be strings")
            # verify the field name is ascii
            try:
                field_name.encode("ascii")
            except UnicodeEncodeError:
                raise TypeError("non-ascii field name not allowed")

            dt = createDataType(field["type"])  # recursive call
            if dt is None:
                raise Exception("unexpected error")
            subtypes.append((field["name"], dt))  # append tuple

        dtRet = np.dtype(subtypes)
    else:
        dtRet = createBaseDataType(typeItem)  # create non-compound dt
    return dtRet


def validateTypeItem(typeItem):
    """
    Validate a json type - call createDataType and if no exception,
       it's valid
    """
    createDataType(typeItem)
    # throws KeyError, TypeError, or ValueError


def getBaseTypeJson(type_name):
    """
    Return JSON representation of a predefined type string
    """
    predefined_int_types = (
        "H5T_STD_I8",
        "H5T_STD_U8",
        "H5T_STD_I16",
        "H5T_STD_U16",
        "H5T_STD_I32",
        "H5T_STD_U32",
        "H5T_STD_I64",
        "H5T_STD_U64",
    )
    predefined_float_types = ("H5T_IEEE_F16", "H5T_IEEE_F32", "H5T_IEEE_F64")
    type_json = {}
    # predefined typenames start with 'H5T' and end with "LE" or "BE"
    if all(
        (
            type_name.startswith("H5T_"),
            type_name[-1] == "E",
            type_name[-2] in ("L", "B"),
        )
    ):
        # trime of the "BE/"LE"
        type_prefix = type_name[:-2]
        if type_prefix in predefined_int_types:
            type_json["class"] = "H5T_INTEGER"
            type_json["base"] = type_name
        elif type_prefix in predefined_float_types:
            type_json["class"] = "H5T_FLOAT"
            type_json["base"] = type_name
        else:
            raise TypeError("Invalid type name")
    else:
        raise TypeError("Invalid type name")
    return type_json


def getSubType(dt_parent, fields):
    """ Return a dtype that is a compound type composed of
        the fields given in the field_names list
    """
    if len(dt_parent) == 0:
        raise TypeError("getSubType - parent must be compound type")
    if not fields:
        raise TypeError("null field specification")
    if isinstance(fields, str):
        fields = [fields,]  # convert to a list

    field_names = set(dt_parent.names)
    dt_items = []
    for field in fields:
        if field not in field_names:
            raise TypeError(f"field: {field} is not defined in parent type")
        dt_items.append((field, dt_parent[field]))
    dt = np.dtype(dt_items)

    return dt
