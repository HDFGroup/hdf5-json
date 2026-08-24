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
import time

from h5json.time_util import unixTimeToUTC, elapsedTime


class TimeUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TimeUtilTest, self).__init__(*args, **kwargs)
        # main
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testUnixTimeToUTC(self):
        # unix epoch
        self.assertEqual(unixTimeToUTC(0), "1970-01-01T00:00:00Z")

        # arbitrary known timestamps
        self.assertEqual(unixTimeToUTC(1414643121), "2014-10-30T04:25:21Z")
        self.assertEqual(unixTimeToUTC(1000000000), "2001-09-09T01:46:40Z")

        # result should be a string ending in "Z" (not "+00:00")
        iso_str = unixTimeToUTC(1600000000)
        self.assertTrue(isinstance(iso_str, str))
        self.assertTrue(iso_str.endswith("Z"))
        self.assertFalse("+" in iso_str)

    def testElapsedTimeInvalid(self):
        # a timestamp in the future is invalid
        future_ts = int(time.time()) + 1000
        self.assertEqual(elapsedTime(future_ts), "Invalid timestamp!")

    def testElapsedTimeSeconds(self):
        # a timestamp of "now" should just report seconds
        now_ts = int(time.time())
        result = elapsedTime(now_ts)
        self.assertTrue(result.endswith("seconds"))
        self.assertFalse("days" in result)
        self.assertFalse("hours" in result)
        self.assertFalse("minutes" in result)

    def testElapsedTimeMinutes(self):
        # 2 minutes, 5 seconds ago
        ts = int(time.time()) - 125
        result = elapsedTime(ts)
        self.assertTrue("2 minutes" in result)
        self.assertTrue("seconds" in result)
        self.assertFalse("days" in result)
        self.assertFalse("hours" in result)

    def testElapsedTimeHours(self):
        # a bit over 2 hours ago
        ts = int(time.time()) - (2 * 60 * 60 + 5 * 60 + 10)
        result = elapsedTime(ts)
        self.assertTrue("hours" in result)
        self.assertTrue("minutes" in result)
        self.assertTrue("seconds" in result)
        self.assertFalse("days" in result)

    def testElapsedTimeDays(self):
        # a bit over 2 days ago
        ts = int(time.time()) - (2 * 24 * 60 * 60 + 60 * 60 + 60 + 1)
        result = elapsedTime(ts)
        self.assertTrue("days" in result)
        self.assertTrue("hours" in result)
        self.assertTrue("minutes" in result)
        self.assertTrue("seconds" in result)


if __name__ == "__main__":
    # setup test files

    unittest.main()
