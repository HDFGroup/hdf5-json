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
from h5json.h5reader import H5NullReader
from h5json.h5writer import H5NullWriter
from h5json.objid import isRootObjId, isSchema2Id


class FakeDb:
    """ minimal stand-in for Hdf5db, just enough to support weakref and
    the .root_id attribute the readers look at during open() """

    def __init__(self, root_id=None):
        self.root_id = root_id


class H5ReaderTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(H5ReaderTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        # create logger

        handler = logging.FileHandler("./h5reader_test.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        self.log.info("init!")

    # --- H5NullReader tests (instantiated directly) ---

    def testH5NullReaderInit(self):
        reader = H5NullReader(None, app_logger=self.log)
        self.assertIsNone(reader.filepath)
        self.assertIsNone(reader.get_root_id())
        self.assertTrue(reader.isClosed())
        self.assertTrue(reader.closed)  # property delegates to isClosed()

        reader2 = H5NullReader("some/file.h5", app_logger=self.log)
        self.assertEqual(reader2.filepath, "some/file.h5")

    def testH5NullReaderGetAttribute(self):
        # H5NullReader.getAttribute always returns None regardless of input
        reader = H5NullReader(None, app_logger=self.log)
        self.assertIsNone(reader.getAttribute("g-not-a-real-id", "attr1"))
        self.assertIsNone(reader.getAttribute("g-not-a-real-id", "attr1", includeData=False))

    def testH5NullReaderGetDatasetValues(self):
        # H5NullReader.getDatasetValues always returns None
        reader = H5NullReader(None, app_logger=self.log)
        self.assertIsNone(reader.getDatasetValues("d-not-a-real-id"))
        self.assertIsNone(reader.getDatasetValues("d-not-a-real-id", sel=None, dtype=None))

    def testH5NullReaderGetStats(self):
        reader = H5NullReader(None, app_logger=self.log)
        stats = reader.getStats()
        self.assertEqual(set(stats.keys()), {"created", "lastModified", "owner"})
        self.assertEqual(stats["created"], 0)
        self.assertEqual(stats["lastModified"], 0)
        self.assertEqual(stats["owner"], "")

    def testH5NullReaderGetObjectById(self):
        # root_id starts out as None; a request for that same id returns a
        # bare root group, anything else raises KeyError
        reader = H5NullReader(None, app_logger=self.log)
        root_json = reader.getObjectById(None)
        self.assertEqual(root_json["links"], {})
        self.assertEqual(root_json["attributes"], {})
        self.assertEqual(root_json["cpl"], {})
        self.assertTrue("created" in root_json)

        with self.assertRaises(KeyError):
            reader.getObjectById("g-some-other-id")

    def testH5NullReaderOpenClose(self):
        # exercise open()/close()/isClosed() directly, with a db set via
        # set_db() the way Hdf5db does internally
        reader = H5NullReader(None, app_logger=self.log)
        fake_db = FakeDb(root_id=None)
        reader.set_db(fake_db)
        self.assertTrue(reader.closed)

        root_id = reader.open()
        self.assertIsNotNone(root_id)
        self.assertFalse(reader.closed)
        self.assertEqual(reader.get_root_id(), root_id)

        # calling open() again while already open is a no-op that returns
        # the same root_id
        root_id_again = reader.open()
        self.assertEqual(root_id_again, root_id)

        reader.close()
        self.assertTrue(reader.closed)
        self.assertTrue(reader.isClosed())

    def testH5NullReaderOpenUsesDbRootId(self):
        # if the db already has a root_id set, open() should adopt it
        # rather than mint a new one
        reader = H5NullReader(None, app_logger=self.log)
        fake_db = FakeDb(root_id="g-deadbeefdeadbeefdeadbeefdeadbeef")
        reader.set_db(fake_db)
        root_id = reader.open()
        self.assertEqual(root_id, "g-deadbeefdeadbeefdeadbeefdeadbeef")

    def testH5NullReaderOpenNoDb(self):
        # if the weakref'd db has gone away (or was never usable), open()
        # should raise IOError/ValueError rather than silently proceeding
        reader = H5NullReader(None, app_logger=self.log)
        fake_db = FakeDb()
        reader.set_db(fake_db)
        del fake_db
        gc.collect()
        with self.assertRaises(ValueError):
            reader.open()

    # --- H5Reader base class concrete methods (exercised via H5NullReader,
    # which inherits them unchanged) ---

    def testSetDbAndDbProperty(self):
        reader = H5NullReader(None, app_logger=self.log)
        fake_db = FakeDb(root_id=None)
        reader.set_db(fake_db)
        self.assertIs(reader.db, fake_db)

    def testFilepathProperty(self):
        reader = H5NullReader("myfile.h5", app_logger=self.log)
        self.assertEqual(reader.filepath, "myfile.h5")

    def testClosedProperty(self):
        reader = H5NullReader(None, app_logger=self.log)
        # never opened - closed should be True
        self.assertTrue(reader.closed)
        fake_db = FakeDb(root_id=None)
        reader.set_db(fake_db)
        reader.open()
        self.assertFalse(reader.closed)
        reader.close()
        self.assertTrue(reader.closed)

    def testQueryDatasetNotImplemented(self):
        reader = H5NullReader(None, app_logger=self.log)
        with self.assertRaises(NotImplementedError):
            reader.queryDataset("d-some-id", "field('_') > 0")

    # --- H5NullReader/H5NullWriter as installed automatically by Hdf5db.open() ---

    def testHdf5dbDefaultsToNullReaderWriter(self):
        db = Hdf5db(app_logger=self.log)
        self.assertIsNone(db.reader)
        self.assertIsNone(db.writer)

        root_id = db.open()
        self.assertIsInstance(db.reader, H5NullReader)
        self.assertIsInstance(db.writer, H5NullWriter)
        self.assertTrue(isSchema2Id(root_id))
        self.assertTrue(isRootObjId(root_id))
        self.assertFalse(db.closed)

        # root group should be readable and start out empty
        root_json = db.getObjectById(root_id)
        self.assertEqual(root_json["links"], {})
        self.assertEqual(root_json["attributes"], {})

        # H5NullReader.getAttribute() always returns None - exercise it
        # directly via db.reader, since Hdf5db.getAttribute() itself
        # resolves attributes from the in-memory db, not the reader
        self.assertIsNone(db.reader.getAttribute(root_id, "not_an_attr"))

        # H5NullReader.getDatasetValues() always returns None
        self.assertIsNone(db.reader.getDatasetValues("d-not-a-real-id"))

        db.close()
        self.assertTrue(db.closed)

        # re-opening should give back the same root_id
        obj_id = db.open()
        self.assertEqual(obj_id, root_id)
        db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
