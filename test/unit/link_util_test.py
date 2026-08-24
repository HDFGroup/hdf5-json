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
import uuid

from h5json.link_util import validateLinkName, getLinkClass, getLinkId, getLinkPath
from h5json.link_util import getLinkFilePath, isEqualLink, h5Join


class LinkUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LinkUtilTest, self).__init__(*args, **kwargs)
        # main
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testValidateLinkName(self):
        # valid names should not raise
        validateLinkName("mylink")
        validateLinkName("link_with_underscore")
        validateLinkName("")

        # non-string names are invalid
        with self.assertRaises(ValueError):
            validateLinkName(42)
        with self.assertRaises(ValueError):
            validateLinkName(None)

        # names with a slash are invalid
        with self.assertRaises(ValueError):
            validateLinkName("foo/bar")

    def testGetLinkClassHard(self):
        tgt_id = str(uuid.uuid4())

        # explicit class
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        self.assertEqual(getLinkClass(link_json), "H5L_TYPE_HARD")

        # class can be inferred from presence of "id"
        link_json = {"id": tgt_id}
        self.assertEqual(getLinkClass(link_json), "H5L_TYPE_HARD")

        # invalid id
        link_json = {"class": "H5L_TYPE_HARD", "id": "not-a-uuid"}
        with self.assertRaises(ValueError):
            getLinkClass(link_json)

        # mismatched class
        link_json = {"class": "H5L_TYPE_SOFT", "id": tgt_id}
        with self.assertRaises(ValueError):
            getLinkClass(link_json)

    def testGetLinkClassSoft(self):
        # explicit class
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": "/g1/g2"}
        self.assertEqual(getLinkClass(link_json), "H5L_TYPE_SOFT")

        # class can be inferred from h5path (no file/h5domain)
        link_json = {"h5path": "/g1/g2"}
        self.assertEqual(getLinkClass(link_json), "H5L_TYPE_SOFT")

        # mismatched class
        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1/g2"}
        with self.assertRaises(ValueError):
            getLinkClass(link_json)

    def testGetLinkClassExternal(self):
        # explicit class, "file" key
        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1", "file": "other.h5"}
        self.assertEqual(getLinkClass(link_json), "H5L_TYPE_EXTERNAL")

        # class inferred from "file"
        link_json = {"h5path": "/g1", "file": "other.h5"}
        self.assertEqual(getLinkClass(link_json), "H5L_TYPE_EXTERNAL")

        # class inferred from deprecated "h5domain"
        link_json = {"h5path": "/g1", "h5domain": "other.h5"}
        self.assertEqual(getLinkClass(link_json), "H5L_TYPE_EXTERNAL")

        # mismatched class
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": "/g1", "file": "other.h5"}
        with self.assertRaises(ValueError):
            getLinkClass(link_json)

    def testGetLinkClassErrors(self):
        tgt_id = str(uuid.uuid4())

        # both id and h5path set
        link_json = {"id": tgt_id, "h5path": "/g1"}
        with self.assertRaises(ValueError):
            getLinkClass(link_json)

        # neither id nor h5path set
        link_json = {"class": "H5L_TYPE_HARD"}
        with self.assertRaises(ValueError):
            getLinkClass(link_json)

        link_json = {}
        with self.assertRaises(ValueError):
            getLinkClass(link_json)

    def testGetLinkId(self):
        tgt_id = str(uuid.uuid4())
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        self.assertEqual(getLinkId(link_json), tgt_id)

        # non-hard links should raise TypeError
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": "/g1"}
        with self.assertRaises(TypeError):
            getLinkId(link_json)

        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1", "file": "other.h5"}
        with self.assertRaises(TypeError):
            getLinkId(link_json)

    def testGetLinkPath(self):
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": "/g1/g2"}
        self.assertEqual(getLinkPath(link_json), "/g1/g2")

        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1/g2", "file": "other.h5"}
        self.assertEqual(getLinkPath(link_json), "/g1/g2")

        # hard links should raise TypeError
        tgt_id = str(uuid.uuid4())
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        with self.assertRaises(TypeError):
            getLinkPath(link_json)

    def testGetLinkFilePath(self):
        # standard "file" key
        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1", "file": "other.h5"}
        self.assertEqual(getLinkFilePath(link_json), "other.h5")

        # deprecated "h5domain" key for backward compatibility
        link_json = {"h5path": "/g1", "h5domain": "other.h5"}
        self.assertEqual(getLinkFilePath(link_json), "other.h5")

        # non-external links should raise TypeError
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": "/g1"}
        with self.assertRaises(TypeError):
            getLinkFilePath(link_json)

        tgt_id = str(uuid.uuid4())
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        with self.assertRaises(TypeError):
            getLinkFilePath(link_json)

    def testIsEqualLinkHard(self):
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())

        link1 = {"class": "H5L_TYPE_HARD", "id": id1}
        link2 = {"class": "H5L_TYPE_HARD", "id": id1}
        self.assertTrue(isEqualLink(link1, link2))

        link3 = {"class": "H5L_TYPE_HARD", "id": id2}
        self.assertFalse(isEqualLink(link1, link3))

    def testIsEqualLinkSoft(self):
        link1 = {"class": "H5L_TYPE_SOFT", "h5path": "/g1/g2"}
        link2 = {"class": "H5L_TYPE_SOFT", "h5path": "/g1/g2"}
        self.assertTrue(isEqualLink(link1, link2))

        link3 = {"class": "H5L_TYPE_SOFT", "h5path": "/g1/g3"}
        self.assertFalse(isEqualLink(link1, link3))

    def testIsEqualLinkExternal(self):
        link1 = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1", "file": "other.h5"}
        link2 = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1", "file": "other.h5"}
        self.assertTrue(isEqualLink(link1, link2))

        link3 = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g1", "file": "different.h5"}
        self.assertFalse(isEqualLink(link1, link3))

        link4 = {"class": "H5L_TYPE_EXTERNAL", "h5path": "/g2", "file": "other.h5"}
        self.assertFalse(isEqualLink(link1, link4))

    def testIsEqualLinkDifferentClasses(self):
        tgt_id = str(uuid.uuid4())
        hard_link = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        soft_link = {"class": "H5L_TYPE_SOFT", "h5path": "/g1"}
        self.assertFalse(isEqualLink(hard_link, soft_link))

    def testIsEqualLinkErrors(self):
        tgt_id = str(uuid.uuid4())
        hard_link = {"class": "H5L_TYPE_HARD", "id": tgt_id}

        # non-dict argument
        with self.assertRaises(TypeError):
            isEqualLink(hard_link, "not a dict")
        with self.assertRaises(TypeError):
            isEqualLink("not a dict", hard_link)

        # missing "class" key
        with self.assertRaises(TypeError):
            isEqualLink(hard_link, {"id": tgt_id})
        with self.assertRaises(TypeError):
            isEqualLink({"id": tgt_id}, hard_link)

    def testH5Join(self):
        self.assertEqual(h5Join("/", "foo"), "/foo")
        self.assertEqual(h5Join("/foo", "bar"), "/foo/bar")
        self.assertEqual(h5Join("/foo/", "bar"), "/foo/bar")
        self.assertEqual(h5Join("/foo", ["bar", "baz"]), "/foo/bar/baz")
        self.assertEqual(h5Join("/foo", ("bar", "baz")), "/foo/bar/baz")

        # no paths to append just returns the original path
        self.assertEqual(h5Join("/foo", None), "/foo")
        self.assertEqual(h5Join("/foo", []), "/foo")
        self.assertEqual(h5Join("/foo", ()), "/foo")


if __name__ == "__main__":
    # setup test files

    unittest.main()
