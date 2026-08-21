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

from h5json.track_util import getTrackTimes


class TrackUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TrackUtilTest, self).__init__(*args, **kwargs)
        # main
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testNoCreationProperties(self):
        self.assertIsNone(getTrackTimes({}))
        self.assertIsNone(getTrackTimes({"creationProperties": {}}))

    def testTrackTimes(self):
        self.assertTrue(getTrackTimes({"creationProperties": {"trackTimes": True}}))
        self.assertFalse(getTrackTimes({"creationProperties": {"trackTimes": False}}))

    def testBareCplDict(self):
        # a bare cpl dict (not wrapped in "creationProperties") is also accepted
        self.assertTrue(getTrackTimes({"trackTimes": True}))

    def testNonBoolValueCoercedToBool(self):
        self.assertTrue(getTrackTimes({"creationProperties": {"trackTimes": 1}}))
        self.assertFalse(getTrackTimes({"creationProperties": {"trackTimes": 0}}))


if __name__ == "__main__":
    unittest.main()
