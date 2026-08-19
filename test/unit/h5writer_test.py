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
    the .root_id attribute the writer looks at during open() """

    def __init__(self, root_id=None):
        self.root_id = root_id


class H5WriterTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(H5WriterTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        # create logger

        handler = logging.FileHandler("./h5writer_test.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        self.log.info("init!")

    # --- H5NullWriter tests (instantiated directly) ---

    def testH5NullWriterInit(self):
        writer = H5NullWriter(None, app_logger=self.log)
        self.assertIsNone(writer.filepath)
        self.assertTrue(writer.isClosed())
        self.assertTrue(writer.closed)  # property delegates to isClosed()
        self.assertIsNone(writer.lastModified)
        self.assertFalse(writer.append)
        self.assertFalse(writer.no_data)

        writer2 = H5NullWriter("some/file.h5", no_data=True, app_logger=self.log)
        self.assertEqual(writer2.filepath, "some/file.h5")
        self.assertTrue(writer2.no_data)
        self.assertFalse(writer2.append)

    def testH5NullWriterAppendNotSupported(self):
        # documented: append=True raises IOError for H5NullWriter
        with self.assertRaises(IOError):
            H5NullWriter(None, append=True, app_logger=self.log)

    def testH5NullWriterGetStats(self):
        writer = H5NullWriter(None, app_logger=self.log)
        stats = writer.getStats()
        self.assertEqual(set(stats.keys()), {"created", "lastModified", "owner"})
        self.assertEqual(stats["created"], 0)
        self.assertEqual(stats["lastModified"], 0)
        self.assertEqual(stats["owner"], "")

    def testH5NullWriterGetFilters(self):
        writer = H5NullWriter(None, app_logger=self.log)
        self.assertEqual(writer.getFilters(), ())
        self.assertEqual(writer.getFilters(compressors_only=True), ())

    def testH5NullWriterOpenCloseFlush(self):
        writer = H5NullWriter(None, app_logger=self.log)
        fake_db = FakeDb(root_id=None)
        writer.set_db(fake_db)
        self.assertTrue(writer.closed)

        root_id = writer.open()
        self.assertIsNotNone(root_id)
        self.assertFalse(writer.closed)

        # calling open() again while already open is a no-op that returns
        # the same root_id
        root_id_again = writer.open()
        self.assertEqual(root_id_again, root_id)

        # H5NullWriter can't actually persist anything - flush() always
        # returns False
        self.assertFalse(writer.flush())

        writer.close()
        self.assertTrue(writer.closed)
        self.assertTrue(writer.isClosed())

    def testH5NullWriterOpenUsesDbRootId(self):
        # if the db already has a root_id set, open() should adopt it
        # rather than mint a new one
        writer = H5NullWriter(None, app_logger=self.log)
        fake_db = FakeDb(root_id="g-deadbeefdeadbeefdeadbeefdeadbeef")
        writer.set_db(fake_db)
        root_id = writer.open()
        self.assertEqual(root_id, "g-deadbeefdeadbeefdeadbeefdeadbeef")

    def testH5NullWriterOpenNoDb(self):
        # if the weakref'd db has gone away, open() should raise
        # ValueError rather than silently proceeding
        writer = H5NullWriter(None, app_logger=self.log)
        fake_db = FakeDb()
        writer.set_db(fake_db)
        del fake_db
        gc.collect()
        with self.assertRaises(ValueError):
            writer.open()

    # --- H5Writer base class concrete methods (exercised via H5NullWriter,
    # which inherits them unchanged) ---

    def testSetDbAndDbProperty(self):
        writer = H5NullWriter(None, app_logger=self.log)
        fake_db = FakeDb(root_id=None)
        writer.set_db(fake_db)
        self.assertIs(writer.db, fake_db)

    def testDbPropertyBeforeSetDb(self):
        # documented: db property should return None (with a debug log)
        # rather than raise, when no db has been set yet
        writer = H5NullWriter(None, app_logger=self.log)
        self.assertIsNone(writer.db)

    def testFilepathProperty(self):
        writer = H5NullWriter("myfile.h5", app_logger=self.log)
        self.assertEqual(writer.filepath, "myfile.h5")

    def testAppendAndNoDataProperties(self):
        writer = H5NullWriter("myfile.h5", no_data=True, app_logger=self.log)
        self.assertFalse(writer.append)
        self.assertTrue(writer.no_data)

    def testLastModifiedProperty(self):
        writer = H5NullWriter(None, app_logger=self.log)
        # never flushed/written - lastModified should be None
        self.assertIsNone(writer.lastModified)

    def testClosedProperty(self):
        writer = H5NullWriter(None, app_logger=self.log)
        # never opened - closed should be True
        self.assertTrue(writer.closed)
        fake_db = FakeDb(root_id=None)
        writer.set_db(fake_db)
        writer.open()
        self.assertFalse(writer.closed)
        writer.close()
        self.assertTrue(writer.closed)

    def testQueryDatasetNotImplemented(self):
        writer = H5NullWriter(None, app_logger=self.log)
        with self.assertRaises(NotImplementedError):
            writer.queryDataset("d-some-id", "field('_') > 0")

    # --- H5NullReader/H5NullWriter as installed automatically by Hdf5db.open() ---

    def testHdf5dbDefaultsToNullWriter(self):
        db = Hdf5db(app_logger=self.log)
        self.assertIsNone(db.reader)
        self.assertIsNone(db.writer)

        root_id = db.open()
        self.assertIsInstance(db.reader, H5NullReader)
        self.assertIsInstance(db.writer, H5NullWriter)
        self.assertTrue(isSchema2Id(root_id))
        self.assertTrue(isRootObjId(root_id))
        self.assertFalse(db.closed)

        # Hdf5db.close() special-cases H5NullWriter: since there's nothing
        # to persist, it skips flush()/writer.close() entirely and only
        # closes the reader - db.closed still reports True because it
        # checks the reader first.
        db.close()
        self.assertTrue(db.closed)
        self.assertTrue(db.reader.isClosed())
        self.assertFalse(db.writer.isClosed())

        # re-opening should give back the same root_id
        obj_id = db.open()
        self.assertEqual(obj_id, root_id)
        db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
