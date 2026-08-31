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

from datetime import datetime
import time
import os
import pytz


def unixTimeToUTC(timestamp):
    """Convert unix timestamp (seconds since Jan 1, 1970, to ISO-8601
    compatible UTC time string.

    """
    utc = pytz.utc
    dtTime = datetime.fromtimestamp(timestamp, utc)
    iso_str = dtTime.isoformat()
    # isoformat returns a string like this:
    # '2014-10-30T04:25:21+00:00'
    # strip off the '+00:00' and replace
    # with 'Z' (both are ISO-8601 compatible)
    npos = iso_str.rfind("+")
    iso_z = iso_str[:npos] + "Z"
    return iso_z


def elapsedTime(timestamp):
    """Get Elapsed time from given timestamp"""
    delta = int(time.time()) - timestamp
    if delta < 0:
        return "Invalid timestamp!"
    day_length = 24 * 60 * 60
    days = 0
    hour_length = 60 * 60
    hours = 0
    minute_length = 60
    minutes = 0
    ret_str = ""

    if delta > day_length:
        days = delta // day_length
        delta = delta % day_length
        ret_str += f"{days} days "
    if delta > hour_length or days > 0:
        hours = delta // hour_length
        delta = delta % hour_length
        ret_str += f"{hours} hours "
    if delta > minute_length or days > 0 or hours > 0:
        minutes = delta // minute_length
        delta = delta % minute_length
        ret_str += f"{minutes} minutes "
    ret_str += f"{delta} seconds"
    return ret_str


# Fallback anchor for getNow() on platforms with a coarse wall clock: a
# wall-clock reading paired with a monotonic reading taken at the same moment,
# so later calls can advance the wall clock by a precisely measured delta.
# Captured once, at import.
_start_time = time.time()
_start_time_relative = time.perf_counter()


def getNow(app=None):
    """
    Get current time in unix timestamp

    Returns a precise timestamp even on platforms where
    time.time() has low resolution (e.g. Windows)
    """
    system = os.name
    current_time = 0

    if system == "nt":
        # Windows: before Python 3.13, time.time() only advances once per
        # ~15.6ms system tick, so objects created in a loop get timestamps that
        # compare equal. Callers order attributes and links by their "created"
        # value, and equal timestamps leave that order undefined - so advance a
        # wall-clock anchor by a perf_counter delta rather than reading the
        # coarse clock directly. Prefer the caller's anchor when it has one; it
        # is set once at node startup, so it stays consistent across a process.
        if app is not None and "start_time_relative" in app and "start_time" in app:
            start_time = app["start_time"]
            start_time_relative = app["start_time_relative"]
        else:
            start_time = _start_time
            start_time_relative = _start_time_relative
        current_time = (time.perf_counter() - start_time_relative) + start_time
    elif system == "posix":
        # Unix - time.time() is already fine-grained here
        current_time = time.time()
    else:
        raise ValueError(f"Unsupported OS: {system}")

    return current_time
