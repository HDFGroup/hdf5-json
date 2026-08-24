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
#
# link_util:
# link related functions
#
from h5json.objid import isValidUuid


def validateLinkName(name):
    """ verify the link name is valid """
    if not isinstance(name, str):
        msg = "Unexpected type for link name"
        raise ValueError(msg)
    if name.find("/") >= 0:
        msg = "link name contains slash"
        raise ValueError(msg)


def getLinkClass(link_json):
    """ verify this is a valid link
        returns the link class """
    if "class" in link_json:
        link_class = link_json["class"]
    else:
        link_class = None
    if "h5path" in link_json and "id" in link_json:
        msg = "link tgt_id and h5path both set"
        raise ValueError(msg)
    if "id" in link_json:
        tgt_id = link_json["id"]
        if not isValidUuid(tgt_id):
            msg = f"link with invalid id: {tgt_id}"
            raise ValueError(msg)
        if link_class:
            if link_class != "H5L_TYPE_HARD":
                msg = f"expected link class to be H5L_TYPE_HARD but got: {link_class}"
                raise ValueError(msg)
        else:
            link_class = "H5L_TYPE_HARD"
    elif link_json.get("h5path"):
        if link_json.get("h5domain") or link_json.get("file"):
            if link_class:
                if link_class != "H5L_TYPE_EXTERNAL":
                    msg = f"expected link class to be H5L_TYPE_EXTERNAL but got: {link_class}"
                    raise ValueError(msg)
            else:
                link_class = "H5L_TYPE_EXTERNAL"
        else:
            if link_class:
                if link_class != "H5L_TYPE_SOFT":
                    msg = f"expected link class to be H5L_TYPE_SOFT but got: {link_class}"
                    raise ValueError(msg)
            else:
                link_class = "H5L_TYPE_SOFT"
    else:
        msg = "link with no id or h5path"
        raise ValueError(msg)

    return link_class


def getLinkId(link_json):
    """ return id for hard links, otherwise raise type error """
    if getLinkClass(link_json) != "H5L_TYPE_HARD":
        raise TypeError("expected hard link")
    return link_json["id"]


def getLinkPath(link_json):
    """ Returns h5path for soft or external link.  Otherwise raise type error """

    if getLinkClass(link_json) not in ("H5L_TYPE_SOFT", "H5L_TYPE_EXTERNAL"):
        raise TypeError("expected soft or external link")

    return link_json["h5path"]


def getLinkFilePath(link_json):
    """ return file path for an external link.  Otherwise raise type error """
    if getLinkClass(link_json) != "H5L_TYPE_EXTERNAL":
        raise TypeError("expected External Link")
    if "file" in link_json:
        link_file = link_json["file"]
    elif "h5domain" in link_json:
        # h5domain was the deprecated storage key
        # check for backward compatibility
        link_file = link_json["h5domain"]
    else:
        raise KeyError("unexpected link format")
    return link_file


def isEqualLink(link1, link2):
    """ Return True if the two links are the same """

    for obj in (link1, link2):
        if not isinstance(obj, dict):
            raise TypeError(f"unexpected type: {type(obj)}")
        if "class" not in obj:
            raise TypeError("expected class key for link")
    link_class = getLinkClass(link1)
    if link_class != getLinkClass(link2):
        return False  # different link types
    if link_class == "H5L_TYPE_HARD":
        if getLinkId(link1) != getLinkId(link2):
            return False
        else:
            return True
    elif link_class == "H5L_TYPE_SOFT":
        if getLinkPath(link1) != getLinkPath(link2):
            return False
        else:
            return True
    elif link_class == "H5L_TYPE_EXTERNAL":
        if getLinkPath(link1) != getLinkPath(link2):
            return False
        if getLinkFilePath(link1) != getLinkFilePath(link2):
            return False
        return True
    else:
        raise TypeError(f"unexpected link class: {link_class}")


def h5Join(path, paths):
    """ join the paths """

    h5path = path
    if not paths:
        return h5path
    if isinstance(paths, str):
        paths = (paths,)
    for s in paths:
        if h5path[-1] != "/":
            h5path += "/"
        h5path += s
    return h5path
