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
import unittest
import logging
import random
import string
import requests
import numpy as np
from h5json import Hdf5db
from h5json.hsdsstore.httpconn import HttpConn
from h5json.hsdsstore.hsds_writer import HSDSWriter
from h5json.hsdsstore.hsds_reader import HSDSReader
from h5json.h5pystore.h5py_reader import H5pyReader
from h5json.hdf5dtype import special_dtype, Reference
from h5json import selections


class HSDSWriterTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(HSDSWriterTest, self).__init__(*args, **kwargs)
        # main
        self.session = requests.Session()

        # create logger
        logfname = "hsds_writer_test.log"
        loglevel = logging.DEBUG
        logging.basicConfig(filename=logfname, format='%(levelname)s %(asctime)s %(message)s', level=loglevel)
        self.log = logging.getLogger()
        self.log.info("init!")

    def testSimple(self):

        domain_path = "hdf5://home/test_user1/test/writer_test.h5"

        db = Hdf5db(app_logger=self.log)
        db.writer = HSDSWriter(domain_path, app_logger=self.log)
        root_id = db.open()

        stats = db.writer.getStats()
        for k in ("created", "lastModified", "owner"):
            self.assertTrue(k in stats)
        http_conn = HttpConn(domain_path, mode='r', retries=1)
        http_conn.open()

        db.createAttribute(root_id, "attr1", value=[1, 2, 3, 4])
        db.createAttribute(root_id, "attr2", 42)

        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        db.createAttribute(g1_id, "a1", "hello")
        g2_id = db.createGroup()
        db.createHardLink(root_id, "g2", g2_id)

        # validate - get the root group and check counts
        http_rsp = http_conn.GET(f"/groups/{root_id}")
        self.assertEqual(http_rsp.status_code, 200)
        root_json = http_rsp.json()
        # attribute count should still be zero (hasn't been flushed yet)
        self.assertEqual(root_json["attributeCount"], 0)
        # same for link count
        self.assertEqual(root_json["linkCount"], 0)
        self.assertTrue(db.writer.lastModified is None)  # no write yet
        db.flush()
        self.assertTrue(db.writer.lastModified > 0)  # timestamp should be updated

        # validate - get the root group again and see if counts are updated
        http_rsp = http_conn.GET(f"/groups/{root_id}")
        self.assertEqual(http_rsp.status_code, 200)
        root_json = http_rsp.json()
        # attribute count should still be zero (hasn't been flushed yet)
        self.assertEqual(root_json["attributeCount"], 2)
        # same for link count
        self.assertEqual(root_json["linkCount"], 2)

        g1_1_id = db.createGroup()
        db.createHardLink(g1_id, "g1.1", g1_1_id)
        dset_111_id = db.createDataset(shape=(10, 10), dtype=np.int32)
        arr = np.zeros((10, 10), dtype=np.int32)
        for i in range(10):
            for j in range(10):
                arr[i, j] = i * j
        sel_all = selections.select((10, 10), ...)
        db.setDatasetValues(dset_111_id, sel_all, arr)
        db.flush()

        # validate - get the dataset and check values
        http_rsp = http_conn.GET(f"/datasets/{dset_111_id}/value")
        self.assertEqual(http_rsp.status_code, 200)
        rsp_json = http_rsp.json()
        self.assertTrue("value" in rsp_json)
        rsp_value = rsp_json["value"]
        self.assertEqual(len(rsp_value), 10)
        for i in range(10):
            row = rsp_value[i]
            self.assertEqual(len(row), 10)
            for j in range(10):
                self.assertEqual(row[j], i * j)

        db.createHardLink(g1_1_id, "dset1.1.1", dset_111_id)
        db.createSoftLink(g2_id, "slink", "somewhere")
        db.createExternalLink(g2_id, "extlink", "somewhere", "someplace")
        db.createCustomLink(g2_id, "cust", {"foo": "bar"})
        db.flush()

        # create a link, then delete before flushing
        tmp_grp_id = db.createGroup("tmp_group")
        db.createHardLink(g1_1_id, "tmp_group", tmp_grp_id)
        db.deleteLink(g1_1_id, "tmp_group")
        db.flush()

        # validate - check that links got updated
        http_rsp = http_conn.GET(f"/groups/{g2_id}/links")
        self.assertEqual(http_rsp.status_code, 200)
        g2links_json = http_rsp.json()
        self.assertTrue("links" in g2links_json)
        g2links = g2links_json["links"]
        self.assertTrue(len(g2links), 2)  # custom link will be ignored

        db.createAttribute(g1_id, "a2", "bye-bye")
        db.flush()

        g21 = db.createGroup()
        db.createHardLink(g2_id, "g2.1", g21)
        db.flush()

        # update one element of the dataset
        sel = selections.select((10, 10), (slice(4, 5), slice(4, 5)))
        arr = np.zeros((), dtype=np.int32)
        arr[()] = 42
        db.setDatasetValues(dset_111_id, sel, arr)
        db.flush()

        # validate - check that just the one element is modified
        http_rsp = http_conn.GET(f"/datasets/{dset_111_id}/value")
        self.assertEqual(http_rsp.status_code, 200)
        rsp_json = http_rsp.json()
        self.assertTrue("value" in rsp_json)
        rsp_value = rsp_json["value"]
        self.assertEqual(len(rsp_value), 10)
        for i in range(10):
            row = rsp_value[i]
            self.assertEqual(len(row), 10)
            for j in range(10):
                if i == 4 and j == 4:
                    expected = 42
                else:
                    expected = i * j
                self.assertEqual(row[j], expected)

        # create a scalar dataset
        dset_112_id = db.createDataset(shape=(), dtype=np.int32)
        arr = np.zeros((), dtype=np.int32)
        arr[()] = 42
        sel_all = selections.select((), ...)
        db.setDatasetValues(dset_112_id, sel_all, arr)
        db.createHardLink(g1_id, "dset1.1.2", dset_112_id)
        db.flush()

        # validate - get the scalar dataset value
        http_rsp = http_conn.GET(f"/datasets/{dset_112_id}/value")
        self.assertEqual(http_rsp.status_code, 200)
        rsp_json = http_rsp.json()
        self.assertTrue("value" in rsp_json)
        rsp_value = rsp_json["value"]
        self.assertEqual(rsp_value, 42)

        # create a dataset and try to read from it
        dset_222_id = db.createDataset(shape=(10, 10), dtype=np.int32)
        sel_all = selections.select((10, 10), ...)
        arr = db.getDatasetValues(dset_222_id, sel_all)
        self.assertTrue((arr == 0).all())

        db.close()

    def testReaderWriter(self):
        # try reading and writer to an HSDS domain
        # create a random string so we don't try to open an existing file
        filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        domain_path = "/home/test_user1/test/" + filename + ".h5"
        db = Hdf5db(app_logger=self.log)
        db.writer = HSDSWriter(domain_path, app_logger=self.log)
        self.assertEqual(db.writer.filepath, domain_path)
        root_id = db.open()
        self.assertTrue(root_id)
        db.reader = HSDSReader(domain_path, app_logger=self.log)
        db.close()
        """
        db.writer = HSDSWriter(domain, **kwargs)
        root_id = db.open()
        db.close()
        # now set the reader
        db.reader = HSDSReader(domain, **kwargs)
        db.open()
        """
        root_id2 = db.open()
        self.assertEqual(root_id, root_id2)
        root_json = db.getObjectById(root_id)
        self.assertTrue("id" not in root_json)
        self.assertTrue("created" in root_json)
        self.assertTrue(root_json["created"] > 0)
        self.assertTrue(db.writer.lastModified is None)  # no flush yet

    def testH5PyToHS(self):
        # test reading from HDF5 file and writing to HSDS

        file_path = "data/hdf5/tall.h5"
        domain_path = "hdf5://home/test_user1/test/hsds_writer_test_tall.h5"

        db = Hdf5db(app_logger=self.log)
        db.reader = H5pyReader(file_path)
        db.writer = HSDSWriter(domain_path)
        root_id = db.open()
        root_json = db.getObjectById(root_id)
        db.flush()

        # validate - get the root group and see if counts are correct
        http_conn = HttpConn(domain_path, mode='r', retries=1)
        http_conn.open()
        http_rsp = http_conn.GET(f"/groups/{root_id}")
        self.assertEqual(http_rsp.status_code, 200)
        root_json = http_rsp.json()
        self.assertEqual(root_json["id"], root_id)
        # attribute count should still be zero (hasn't been flushed yet)
        self.assertEqual(root_json["attributeCount"], 2)
        # same for link count
        self.assertEqual(root_json["linkCount"], 2)

        # get the g1 hard link
        http_rsp = http_conn.GET(f"/groups/{root_id}/links/g1")
        self.assertEqual(http_rsp.status_code, 200)
        rsp_json = http_rsp.json()
        g1_link = rsp_json["link"]
        g1_id = g1_link["id"]

        # get the g1 group json
        http_rsp = http_conn.GET(f"/groups/{g1_id}")
        self.assertEqual(http_rsp.status_code, 200)
        g1_json = http_rsp.json()
        self.assertEqual(g1_json["attributeCount"], 0)
        self.assertEqual(g1_json["linkCount"], 2)

        # get the g1.1 link
        http_rsp = http_conn.GET(f"/groups/{g1_id}/links/g1.1")
        self.assertEqual(http_rsp.status_code, 200)
        rsp_json = http_rsp.json()
        g1_1_link = rsp_json["link"]
        g1_1_id = g1_1_link["id"]

        # Get the g1.1 json
        http_rsp = http_conn.GET(f"/groups/{g1_1_id}")
        self.assertEqual(http_rsp.status_code, 200)
        g1_json = http_rsp.json()
        self.assertEqual(g1_json["attributeCount"], 0)
        self.assertEqual(g1_json["linkCount"], 2)

        # get the dset1.1.1 link
        http_rsp = http_conn.GET(f"/groups/{g1_1_id}/links/dset1.1.1")
        self.assertEqual(http_rsp.status_code, 200)
        rsp_json = http_rsp.json()
        dset1_1_1_link = rsp_json["link"]
        dset1_1_1_id = dset1_1_1_link["id"]

        # get the dset1.1.1 json
        http_rsp = http_conn.GET(f"/datasets/{dset1_1_1_id}")
        self.assertEqual(http_rsp.status_code, 200)
        dset1_1_1_json = http_rsp.json()
        dset1_1_1_shape = dset1_1_1_json["shape"]
        self.assertEqual(dset1_1_1_shape["class"], "H5S_SIMPLE")

        # get the dset1_1_1 data
        http_rsp = http_conn.GET(f"/datasets/{dset1_1_1_id}/value")
        self.assertEqual(http_rsp.status_code, 200)
        rsp_json = http_rsp.json()
        dset1_1_1_value = rsp_json["value"]
        self.assertEqual(len(dset1_1_1_value), 10)
        for i in range(10):
            row = dset1_1_1_value[i]
            self.assertEqual(len(row), 10)
            for j in range(10):
                self.assertEqual(row[j], i * j)

        db.close()


if __name__ == "__main__":
    # setup test files

    unittest.main()
