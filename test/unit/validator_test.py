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
import glob
import json
import logging
import os
import tempfile
import time
import unittest

from h5json import Hdf5db
from h5json.apps.validator import prepare_validator
from h5json.h5pystore.h5py_plugin import H5pyPlugin
from h5json.jsonstore.h5json_plugin import H5JsonPlugin
from h5json.objid import isSchema2Id

# fixture paths below are relative to the repo root
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

OLD_UUID = "a2c7e1b4-0e5c-11e8-9d63-0242ac120002"

log = logging.getLogger("validator_test")
log.addHandler(logging.NullHandler())


class ValidatorTest(unittest.TestCase):

    def setUp(self):
        self.validator = prepare_validator()
        self.log = log

    def soft_link_doc(self, h5path):
        """Return a minimal document whose root group holds one soft link to h5path."""
        return {
            "apiVersion": "1.1.1",
            "root": OLD_UUID,
            "groups": {
                OLD_UUID: {
                    "alias": ["/"],
                    "links": [{"class": "H5L_TYPE_SOFT", "title": "s", "h5path": h5path}],
                }
            },
        }

    def testH5pathPatternIsLinear(self):
        doc = self.soft_link_doc("/" + "a" * 40 + "/")
        start = time.time()
        self.assertFalse(self.validator.is_valid(doc))
        errors = list(self.validator.iter_errors(doc))
        self.assertLess(time.time() - start, 1.0)
        self.assertTrue(errors)

    def testH5pathPatternLanguage(self):
        # the paths the previous pattern, ^(/?[^/]+)+$, accepted and rejected
        for h5path in ("a", "/a", "a/b", "/a/b/c", "/a b/c.d"):
            self.assertTrue(self.validator.is_valid(self.soft_link_doc(h5path)), h5path)
        for h5path in ("", "/", "//", "a/", "/a/", "a//b", "//a"):
            self.assertFalse(self.validator.is_valid(self.soft_link_doc(h5path)), h5path)

    def testH5tojsonOutputValidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "tall.json")
            src_db = Hdf5db(plugin=H5pyPlugin("data/hdf5/tall.h5", read_only=True, app_logger=self.log),
                            app_logger=self.log)
            src_db.open()
            dst_db = Hdf5db(plugin=H5JsonPlugin(json_path, app_logger=self.log), app_logger=self.log)
            dst_db.open()
            src_db.copy(dst_db)
            dst_db.close()
            src_db.close()
            with open(json_path) as f:
                doc = json.load(f)

        self.assertTrue(isSchema2Id("g-" + doc["root"]))
        self.assertTrue(doc["datasets"])
        errors = [str(e) for e in self.validator.iter_errors(doc)]
        self.assertEqual(errors, [])

        # with 2.0 ids, groups are checked too, not skipped
        doc["groups"][doc["root"]]["links"] = 5
        self.assertFalse(self.validator.is_valid(doc))

    def testSampleCorpus(self):
        failed = []
        for json_path in sorted(glob.glob("data/json/*.json")):
            with open(json_path) as f:
                doc = json.load(f)
            if not self.validator.is_valid(doc):
                failed.append(os.path.basename(json_path))
        # the other files with 2.0 ids now pass; sample.json has unrelated schema
        # mismatches (a vlen "size" key, an inline "id" on a committed type)
        self.assertEqual(failed, ["sample.json"])


if __name__ == "__main__":
    unittest.main()
