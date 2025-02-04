##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HDF (HDF5 REST Server) Service, Libraries and      #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
#
# objID:
# id (uuid) related functions
#


import hashlib
import uuid

S3_URI = "s3://"
FILE_URI = "file://"
AZURE_URI = "blob.core.windows.net/"  # preceded with "https://"
UUID_LEN = 36  # length for uuid strings



def _getStorageProtocol(uri):
    """ returns 's3://', 'file://', or 'https://...net/' prefix if present.
    If the prefix is in the form: https://myaccount.blob.core.windows.net/mycontainer
    (references Azure blob storage), return: https://myaccount.blob.core.windows.net/
    otherwise None """

    if not uri:
        protocol = None
    elif uri.startswith(S3_URI):
        protocol = S3_URI
    elif uri.startswith(FILE_URI):
        protocol = FILE_URI
    elif uri.startswith("https://") and uri.find(AZURE_URI) > 0:
        n = uri.find(AZURE_URI) + len(AZURE_URI)
        protocol = uri[:n]
    elif uri.find("://") >= 0:
        raise ValueError(f"storage uri: {uri} not supported")
    else:
        protocol = None
    return protocol


def _getBaseName(uri):
    """ Return the part of the URI after the storage protocol (if any) """

    protocol = _getStorageProtocol(uri)
    if not protocol:
        return uri
    else:
        return uri[len(protocol):]
    
def _getPrefixForCollection(collection):
    """ Return prefix character for given collection type """
    collection = collection.lower()

    if collection in ("group", "groups"):
        return 'g'
    elif collection in ("dataset", "datasets"):
        return 'd'
    elif collection in ("datatype", "datatypes"):
        return 't'
    elif collection in ("chunk", "chunks"):
        return 'c'
    else:
        raise ValueError(f"unexpected collection type: {collection}")


def getIdHash(id):
    """Return md5 prefix based on id value"""
    m = hashlib.new("md5")
    m.update(id.encode("utf8"))
    hexdigest = m.hexdigest()
    return hexdigest[:5]


def isSchema2Id(id):
    """return true if this is a v2 id"""
    # v1 ids are in the standard UUID format: 8-4-4-4-12
    # v2 ids are in the non-standard: 8-8-4-6-6
    parts = id.split("-")
    if len(parts) != 6:
        raise ValueError(f"Unexpected id formation for uuid: {id}")
    if len(parts[2]) == 8:
        return True
    else:
        return False


def getIdHexChars(id):
    """get the hex chars of the given id"""
    if id[0] == "c":
        # don't include chunk index
        index = id.index("_")
        parts = id[0:index].split("-")
    else:
        parts = id.split("-")
    if len(parts) != 6:
        raise ValueError(f"Unexpected id format for uuid: {id}")
    return "".join(parts[1:])


def hexRot(ch):
    """rotate hex character by 8"""
    return format((int(ch, base=16) + 8) % 16, "x")


def isRootObjId(id):
    """returns true if this is a root id (only for v2 schema)"""
    if not isSchema2Id(id):
        raise ValueError("isRootObjId can only be used with v2 ids")
    validateUuid(id)  # will throw ValueError exception if not a objid
    if id[0] != "g":
        return False  # not a group
    token = getIdHexChars(id)
    # root ids will have last 16 chars rotated version of the first 16
    is_root = True
    for i in range(16):
        if token[i] != hexRot(token[i + 16]):
            is_root = False
            break
    return is_root


def getRootObjId(id):
    """returns root id for this objid if this is a root id
    (only for v2 schema)
    """
    if isRootObjId(id):
        return id  # this is the root id
    token = list(getIdHexChars(id))
    # root ids will have last 16 chars rotated version of the first 16
    for i in range(16):
        token[i + 16] = hexRot(token[i])
    token = "".join(token)
    root_id = "g-" + token[0:8] + "-" + token[8:16] + "-" + token[16:20]
    root_id += "-" + token[20:26] + "-" + token[26:32]

    return root_id


def createObjId(obj_type=None, root_id=None):
    """ create a new objid 
    
        if obj_type is None, return just a bare uuid.
        Otherwise a hsds v2 schema obj_id will be created.
        In this case obj_type should be one of "groups",
        "datasets", "datatypes", "chunks".  If rootid is
        None, a root group obj_id will be created.  Otherwise the 
        obj_id will be a an id that has root_id as it's root.  """

    
    prefix = None
    if obj_type is None:
        # just return a regular uuid
        objid = str(uuid.uuid4())
    else:

        prefix = _getPrefixForCollection(obj_type)
        # schema v2
        salt = uuid.uuid4().hex
        # take a hash to randomize the uuid
        token = list(hashlib.sha256(salt.encode()).hexdigest())

        if root_id:
            # replace first 16 chars of token with first 16 chars of root id
            root_hex = getIdHexChars(root_id)
            token[0:16] = root_hex[0:16]
        else:
            if obj_type != "groups":
                raise ValueError("expected 'groups' obj_type for root group id")
            # use only 16 chars, but make it look a 32 char id
            for i in range(16):
                token[16 + i] = hexRot(token[i])
        # format as a string
        token = "".join(token)
        objid = prefix + "-" + token[0:8] + "-" + token[8:16] + "-"
        objid += token[16:20] + "-" + token[20:26] + "-" + token[26:32]

    return objid


def getS3Key(id):
    """Return s3 key for given id.

    For schema v1:
        A md5 prefix is added to the front of the returned key to better
        distribute S3 objects.
    For schema v2:
        The id is converted to the pattern: "db/{rootid[0:16]}" for rootids and
        "db/id[0:16]/{prefix}/id[16-32]" for other ids
        Chunk ids have the chunk index added after the slash:
        "db/id[0:16]/d/id[16:32]/x_y_z

    For domain id's:
        Return a key with the .domain suffix and no preceding slash.
        For non-default buckets, use the format: <bucket_name>/s3_key
        If the id has a storage specifier ("s3://", "file://", etc.)
        include that along with the bucket name. e.g.: "s3://mybucket/a_folder/a_file.h5"
    """

    base_id = _getBaseName(id)  # strip any s3://, etc.
    if base_id.find("/") > 0:
        # a domain id
        domain_suffix = ".domain.json"
        index = base_id.find("/") + 1
        key = base_id[index:]
        if not key.endswith(domain_suffix):
            if key[-1] != "/":
                key += "/"
            key += domain_suffix
    else:
        if isSchema2Id(id):
            # schema v2 id
            hexid = getIdHexChars(id)
            prefix = id[0]  # one of g, d, t, c
            if prefix not in ("g", "d", "t", "c"):
                raise ValueError(f"Unexpected id: {id}")

            if isRootObjId(id):
                key = f"db/{hexid[0:8]}-{hexid[8:16]}"
            else:
                partition = ""
                if prefix == "c":
                    # use 'g' so that chunks will show up under their dataset
                    s3col = "d"
                    n = id.find("-")
                    if n > 1:
                        # extract the partition index if present
                        partition = "p" + id[1:n]
                else:
                    s3col = prefix
                key = f"db/{hexid[0:8]}-{hexid[8:16]}/{s3col}/{hexid[16:20]}"
                key += f"-{hexid[20:26]}-{hexid[26:32]}"
            if prefix == "c":
                if partition:
                    key += "/"
                    key += partition
                # add the chunk coordinate
                index = id.index("_")  # will raise ValueError if not found
                n = index + 1
                coord = id[n:]
                key += "/"
                key += coord
            elif prefix == "g":
                # add key suffix for group
                key += "/.group.json"
            elif prefix == "d":
                # add key suffix for dataset
                key += "/.dataset.json"
            else:
                # add key suffix for datatype
                key += "/.datatype.json"
        else:
            # v1 id
            # schema v1 id
            idhash = getIdHash(id)
            key = f"{idhash}-{id}"

    return key


def getObjId(s3key):
    """Return object id given valid s3key"""
    if all(
        (
            len(s3key) >= 44 and s3key[0:5].isalnum(),
            len(s3key) >= 44 and s3key[5] == "-",
            len(s3key) >= 44 and s3key[6] in ("g", "d", "c", "t"),
        )
    ):
        # v1 obj keys
        objid = s3key[6:]
    elif s3key.endswith("/.domain.json"):
        objid = "/" + s3key[: -(len("/.domain.json"))]
    elif s3key.startswith("db/"):
        # schema v2 object key
        parts = s3key.split("/")
        chunk_coord = ""  # used only for chunk ids
        partition = ""  # likewise
        token = []
        for ch in parts[1]:
            if ch != "-":
                token.append(ch)

        if len(parts) == 3:
            # root id
            # last part should be ".group.json"
            if parts[2] != ".group.json":
                raise ValueError(f"unexpected S3Key: {s3key}")
            # add 16 more chars using rotated version of first 16
            for i in range(16):
                token.append(hexRot(token[i]))
            prefix = "g"
        elif len(parts) == 5:
            # group, dataset, or datatype or chunk
            for ch in parts[3]:
                if ch != "-":
                    token.append(ch)

            if parts[2] == "g" and parts[4] == ".group.json":
                prefix = "g"  # group json
            elif parts[2] == "t" and parts[4] == ".datatype.json":
                prefix = "t"  # datatype json
            elif parts[2] == "d":
                if parts[4] == ".dataset.json":
                    prefix = "d"  # dataset json
                else:
                    # chunk object
                    prefix = "c"
                    chunk_coord = "_" + parts[4]
            else:
                raise ValueError(f"unexpected S3Key: {s3key}")
        elif len(parts) == 6:
            # chunk key with partitioning
            for ch in parts[3]:
                if ch != "-":
                    token.append(ch)
            if parts[2][0] != "d":
                raise ValueError(f"unexpected S3Key: {s3key}")
            prefix = "c"
            partition = parts[4]
            if partition[0] != "p":
                raise ValueError(f"unexpected S3Key: {s3key}")
            partition = partition[1:]  # strip off the p
            chunk_coord = "_" + parts[5]
        else:
            raise ValueError(f"unexpected S3Key: {s3key}")

        token = "".join(token)
        objid = prefix + partition + "-" + token[0:8] + "-" + token[8:16]
        objid += "-" + token[16:20] + "-" + token[20:26] + "-"
        objid += token[26:32] + chunk_coord
    else:
        msg = f"unexpected S3Key: {s3key}"
        raise ValueError(msg)
    return objid


def isS3ObjKey(s3key):
    """ return True if this is a storage key """
    valid = False
    try:
        objid = getObjId(s3key)
        if objid:
            valid = True
    except KeyError:
        pass  # ignore
    except ValueError:
        pass  # ignore
    return valid


def getCollectionForId(obj_id):
    """return groups/datasets/datatypes based on id"""
    if not isinstance(obj_id, str):
        raise ValueError("invalid object id")
    collection = None
    if obj_id.startswith("g-"):
        collection = "groups"
    elif obj_id.startswith("d-"):
        collection = "datasets"
    elif obj_id.startswith("t-"):
        collection = "datatypes"
    else:
        raise ValueError("not a collection id")
    return collection


def validateUuid(id, obj_class=None):
    """ verify the UUID is well-formed 
        schema can be:
           None: expecting ordinary UUID
           "v1": expecting HSDS v1 format
           "v2": expecting HSDS v2 format
        if set obj_class can be one of "groups", "datasets", "datatypes"
    """
    if not isinstance(id, str):
        raise ValueError("Expected string type")
    if len(id) < UUID_LEN:
        raise ValueError("id is too short to be an object identifier")
    if len(id) == UUID_LEN:
        if obj_class:
            # expected a prefix
            raise ValueError(f"obj_id: {id} not valid for collection: {obj_class}") 
    else:
        # does this have a v1 schema hash tag?
        # e.g.: "a49be-g-314d61b8-9954-11e6-a733-3c15c2da029e",
        if id[:5].isalnum() and id[5] == '-':
            id = id[6:]  # trim off the hash tag
        # validate prefix
        if id[0] not in ("g", "d", "t", "c"):
            raise ValueError("Unexpected prefix")
        if id[0] != "c" and id[1] != "-":
            # chunk ids may have a partition index following the c
            raise ValueError("Unexpected prefix")
        if obj_class is not None:
            obj_class = obj_class.lower()
            if id[0] != _getPrefixForCollection(obj_class):
                raise ValueError(f"unexpected object id {id} for collection: {obj_class}")
        if id[0] == "c":
            # trim the type char and any partition id
            n = id.find("-")
            if n == -1:
                raise ValueError("Invalid chunk id")

            # trim the chunk index for chunk ids
            m = id.find("_")
            if m == -1:
                raise ValueError("Invalid chunk id")
            n += 1
            id = "c-" + id[n:m]
        id = id[2:]
    if len(id) != UUID_LEN:
        # id should be 36 now
        raise ValueError("Unexpected id length")

    for ch in id:
        if ch.isalnum():
            continue
        if ch == "-":
            continue
        raise ValueError(f"Unexpected character in uuid: {ch}")


def isValidUuid(id, obj_class=None):
    try:
        validateUuid(id, obj_class)
        return True
    except ValueError:
        return False


def isValidChunkId(id):
    if not isValidUuid(id):
        return False
    if id[0] != "c":
        return False
    return True


def getClassForObjId(id):
    """return domains/chunks/groups/datasets/datatypes based on id"""
    if not isinstance(id, str):
        raise ValueError("Expected string type")
    if len(id) == 0:
        raise ValueError("Empty string")
    if id[0] == "/":
        return "domains"
    if isValidChunkId(id):
        return "chunks"
    else:
        return getCollectionForId(id)


def isObjId(id):
    """return true if uuid or domain"""
    if not isinstance(id, str) or len(id) == 0:
        return False
    if id.find("/") > 0:
        # domain id is any string in the form <bucket_name>/<domain_path>
        return True
    return isValidUuid(id)


def getUuidFromId(id):
    """strip off the type prefix ('g-' or 'd-', or 't-')
    and return the uuid part"""
    if len(id) == UUID_LEN:
        # just a uuid
        return id
    elif len(id) == UUID_LEN + 2:
        # 'g-', 'd-', or 't-' prefix
        return id[2:]
    else:
        raise ValueError(f"Unexpected obj_id: {id}")
    
 
  