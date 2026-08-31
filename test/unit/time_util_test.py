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
from unittest import mock

from h5json import time_util
from h5json.time_util import unixTimeToUTC, elapsedTime, getNow


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


class _FrozenClock:
    """Stand-in for a wall clock too coarse to separate successive calls.

    Models the pre-3.13 Windows case, where time.time() only advances once per
    ~15.6ms system tick: every call made inside one tick returns the same
    value. Returning a constant is that worst case.
    """

    def __init__(self, value=1700000000.0):
        self.value = value

    def __call__(self):
        return self.value


class GetNowTest(unittest.TestCase):
    """Tests for getNow().

    The Windows-specific behavior is exercised by forcing os.name and stubbing
    the wall clock, rather than by relying on a Windows runner. Two reasons: it
    then runs on every platform, and a real-clock test would silently pass on
    Windows/Python 3.13 anyway - 3.13 switched time.time() to
    GetSystemTimePreciseAsFileTime, so the coarse clock this guards against is
    only observable on 3.11/3.12.
    """

    def __init__(self, *args, **kwargs):
        super(GetNowTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testGetNowResolvesSubTickIntervals(self):
        # objects are ordered by their "created" timestamp, so timestamps that
        # compare equal leave creation order undefined. Spin for an interval far
        # shorter than the ~15.6ms coarse tick but far longer than any
        # platform's monotonic-clock resolution, then require that getNow()
        # noticed. Deliberately not "call it N times and count distinct
        # values": that measures loop speed against perf_counter resolution,
        # which varies by platform, where this measures the property at issue.
        with mock.patch.object(time_util.os, "name", "nt"):
            with mock.patch.object(time_util.time, "time", _FrozenClock()):
                before = getNow()
                deadline = time.perf_counter() + 0.001  # 1ms
                while time.perf_counter() < deadline:
                    pass
                after = getNow()

        self.assertGreater(after, before)

    def testGetNowIsMonotonicOnCoarseClock(self):
        # a batch taken back-to-back must never go backwards, whatever the
        # underlying clock resolution turns out to be
        count = 200
        with mock.patch.object(time_util.os, "name", "nt"):
            with mock.patch.object(time_util.time, "time", _FrozenClock()):
                values = [getNow() for _ in range(count)]

        self.assertEqual(len(values), count)
        self.assertEqual(values, sorted(values))
        # reading the coarse clock directly collapses this to exactly 1; the
        # bound stays well clear of how many ties a fast loop may produce on
        # platforms with a coarser monotonic clock (macOS runners hit ~50%)
        self.assertGreaterEqual(len(set(values)), count // 10)

    def testGetNowPrefersAppAnchor(self):
        # a caller-supplied anchor is used in preference to the module one
        app = {"start_time": 1000000.0, "start_time_relative": time.perf_counter()}
        with mock.patch.object(time_util.os, "name", "nt"):
            with mock.patch.object(time_util.time, "time", _FrozenClock()):
                now = getNow(app)
        # elapsed since the anchor was taken is small, so the result should sit
        # just after the anchor's start_time - not at the frozen clock's value
        self.assertGreaterEqual(now, app["start_time"])
        self.assertLess(now, app["start_time"] + 60)

    def testGetNowIgnoresIncompleteAppAnchor(self):
        # a dict missing either key must not be used as an anchor - it has to
        # fall back to the module anchor, not to the coarse clock
        for app in ({}, {"start_time": 1000000.0}, {"start_time_relative": 0.0}):
            with mock.patch.object(time_util.os, "name", "nt"):
                with mock.patch.object(time_util.time, "time", _FrozenClock()):
                    before = getNow(app)
                    deadline = time.perf_counter() + 0.001
                    while time.perf_counter() < deadline:
                        pass
                    after = getNow(app)
            self.assertGreater(after, before, f"app: {app}")

    def testGetNowTracksWallClock(self):
        # on the real platform, getNow() should agree with the wall clock;
        # guards the anchor being mis-derived rather than merely fine-grained
        self.assertAlmostEqual(getNow(), time.time(), delta=5)

    def testGetNowPosixUsesWallClock(self):
        frozen = _FrozenClock()
        with mock.patch.object(time_util.os, "name", "posix"):
            with mock.patch.object(time_util.time, "time", frozen):
                self.assertEqual(getNow(), frozen.value)

    def testGetNowUnsupportedOS(self):
        with mock.patch.object(time_util.os, "name", "java"):
            self.assertRaises(ValueError, getNow)


if __name__ == "__main__":
    # setup test files

    unittest.main()
