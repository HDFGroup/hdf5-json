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
import gc
import unittest
import logging

from h5json import Hdf5db
from h5json.storage_plugin import NullPlugin
from h5json.objid import isRootObjId, isSchema2Id


class FakeDb:
    """ minimal stand-in for Hdf5db, just enough to support weakref and
    the .root_id attribute NullPlugin looks at during open() """

    def __init__(self, root_id=None):
        self.root_id = root_id


class StoragePluginTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(StoragePluginTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        # create logger

        handler = logging.FileHandler("./storage_plugin_test.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        self.log.info("init!")

    # --- NullPlugin tests (instantiated directly) ---

    def testNullPluginInit(self):
        plugin = NullPlugin(None, app_logger=self.log)
        self.assertIsNone(plugin.filepath)
        self.assertIsNone(plugin.get_root_id())
        self.assertTrue(plugin.isClosed())
        self.assertTrue(plugin.closed)  # property delegates to isClosed()
        self.assertIsNone(plugin.lastModified)
        self.assertFalse(plugin.append)
        self.assertFalse(plugin.no_data)

        plugin2 = NullPlugin("some/file.h5", no_data=True, app_logger=self.log)
        self.assertEqual(plugin2.filepath, "some/file.h5")
        self.assertTrue(plugin2.no_data)
        self.assertFalse(plugin2.append)

    def testNullPluginAppendSupported(self):
        # unlike the old H5NullWriter, append=True is accepted (it's
        # meaningless for a no-op plugin, but no longer an error)
        plugin = NullPlugin(None, append=True, app_logger=self.log)
        self.assertTrue(plugin.append)

    def testNullPluginGetAttribute(self):
        # NullPlugin.getAttribute always returns None regardless of input
        plugin = NullPlugin(None, app_logger=self.log)
        self.assertIsNone(plugin.getAttribute("g-not-a-real-id", "attr1"))
        self.assertIsNone(plugin.getAttribute("g-not-a-real-id", "attr1", includeData=False))

    def testNullPluginGetDatasetValues(self):
        # NullPlugin.getDatasetValues always returns None
        plugin = NullPlugin(None, app_logger=self.log)
        self.assertIsNone(plugin.getDatasetValues("d-not-a-real-id"))
        self.assertIsNone(plugin.getDatasetValues("d-not-a-real-id", sel=None, dtype=None))

    def testNullPluginGetStats(self):
        plugin = NullPlugin(None, app_logger=self.log)
        stats = plugin.getStats()
        self.assertEqual(set(stats.keys()), {"created", "lastModified", "owner"})
        self.assertEqual(stats["created"], 0)
        self.assertEqual(stats["lastModified"], 0)
        self.assertEqual(stats["owner"], "")

    def testNullPluginGetFilters(self):
        plugin = NullPlugin(None, app_logger=self.log)
        self.assertEqual(plugin.getFilters(), ())
        self.assertEqual(plugin.getFilters(compressors_only=True), ())

    def testNullPluginGetObjectById(self):
        # root_id starts out as None; a request for that same id returns a
        # bare root group, anything else raises KeyError
        plugin = NullPlugin(None, app_logger=self.log)
        root_json = plugin.getObjectById(None)
        self.assertEqual(root_json["links"], {})
        self.assertEqual(root_json["attributes"], {})
        self.assertEqual(root_json["creationProperties"], {})
        self.assertTrue("created" in root_json)

        with self.assertRaises(KeyError):
            plugin.getObjectById("g-some-other-id")

    def testNullPluginOpenCloseFlush(self):
        plugin = NullPlugin(None, app_logger=self.log)
        fake_db = FakeDb(root_id=None)
        plugin.set_db(fake_db)
        self.assertTrue(plugin.closed)

        root_id = plugin.open()
        self.assertIsNotNone(root_id)
        self.assertFalse(plugin.closed)

        # calling open() again while already open is a no-op that returns
        # the same root_id
        root_id_again = plugin.open()
        self.assertEqual(root_id_again, root_id)

        # NullPlugin can't actually persist anything - flush() always
        # returns False
        self.assertFalse(plugin.flush())

        plugin.close()
        self.assertTrue(plugin.closed)
        self.assertTrue(plugin.isClosed())

    def testNullPluginOpenUsesDbRootId(self):
        # if the db already has a root_id set, open() should adopt it
        # rather than mint a new one
        plugin = NullPlugin(None, app_logger=self.log)
        fake_db = FakeDb(root_id="g-deadbeefdeadbeefdeadbeefdeadbeef")
        plugin.set_db(fake_db)
        root_id = plugin.open()
        self.assertEqual(root_id, "g-deadbeefdeadbeefdeadbeefdeadbeef")

    def testNullPluginOpenNoDb(self):
        # if the weakref'd db has gone away (or was never usable), open()
        # should raise ValueError rather than silently proceeding
        plugin = NullPlugin(None, app_logger=self.log)
        fake_db = FakeDb()
        plugin.set_db(fake_db)
        del fake_db
        gc.collect()
        with self.assertRaises(ValueError):
            plugin.open()

    # --- StoragePlugin base class concrete methods (exercised via
    # NullPlugin, which inherits them unchanged) ---

    def testSetDbAndDbProperty(self):
        plugin = NullPlugin(None, app_logger=self.log)
        fake_db = FakeDb(root_id=None)
        plugin.set_db(fake_db)
        self.assertIs(plugin.db, fake_db)

    def testFilepathProperty(self):
        plugin = NullPlugin("myfile.h5", app_logger=self.log)
        self.assertEqual(plugin.filepath, "myfile.h5")

    def testAppendAndNoDataProperties(self):
        plugin = NullPlugin("myfile.h5", no_data=True, app_logger=self.log)
        self.assertFalse(plugin.append)
        self.assertTrue(plugin.no_data)

    def testLastModifiedProperty(self):
        plugin = NullPlugin(None, app_logger=self.log)
        # never flushed/written - lastModified should be None
        self.assertIsNone(plugin.lastModified)

    def testClosedProperty(self):
        plugin = NullPlugin(None, app_logger=self.log)
        # never opened - closed should be True
        self.assertTrue(plugin.closed)
        fake_db = FakeDb(root_id=None)
        plugin.set_db(fake_db)
        plugin.open()
        self.assertFalse(plugin.closed)
        plugin.close()
        self.assertTrue(plugin.closed)

    def testQueryDatasetNotImplemented(self):
        plugin = NullPlugin(None, app_logger=self.log)
        with self.assertRaises(NotImplementedError):
            plugin.queryDataset("d-some-id", "field('_') > 0")

    # --- NullPlugin as installed automatically by Hdf5db.open() ---

    def testHdf5dbDefaultsToNullPlugin(self):
        db = Hdf5db(app_logger=self.log)
        self.assertIsNone(db.plugin)

        root_id = db.open()
        self.assertIsInstance(db.plugin, NullPlugin)
        self.assertTrue(isSchema2Id(root_id))
        self.assertTrue(isRootObjId(root_id))
        self.assertFalse(db.closed)

        # root group should be readable and start out empty
        root_json = db.getObjectById(root_id)
        self.assertEqual(root_json["links"], {})
        self.assertEqual(root_json["attributes"], {})

        # NullPlugin.getAttribute() always returns None - exercise it
        # directly via db.plugin, since Hdf5db.getAttribute() itself
        # resolves attributes from the in-memory db, not the plugin
        self.assertIsNone(db.plugin.getAttribute(root_id, "not_an_attr"))

        # NullPlugin.getDatasetValues() always returns None
        self.assertIsNone(db.plugin.getDatasetValues("d-not-a-real-id"))

        db.close()
        # a single plugin now serves both reads and writes, so close()
        # always closes it (unlike the old reader/writer split, where
        # Hdf5db.close() special-cased H5NullWriter and left it unclosed)
        self.assertTrue(db.closed)
        self.assertTrue(db.plugin.isClosed())

        # re-opening should give back the same root_id
        obj_id = db.open()
        self.assertEqual(obj_id, root_id)
        db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
