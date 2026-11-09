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
import logging
import os
import sys

LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"

# C0 controls, DEL and C1 controls -> printable escapes, so a logged name can't start a new line
_CONTROL_ESCAPES = {c: f"\\x{c:02x}" for c in (*range(0x20), *range(0x7f, 0xa0))}
_CONTROL_ESCAPES.update({ord("\n"): "\\n", ord("\r"): "\\r", ord("\t"): "\\t"})


class EscapingFormatter(logging.Formatter):
    """Formatter that escapes control characters in the log message."""

    def formatMessage(self, record):
        record = logging.makeLogRecord(record.__dict__)
        record.message = record.message.translate(_CONTROL_ESCAPES)
        return super().formatMessage(record)


def _open_log_file(logfname):
    """Open logfname for appending without following a symlink; None if unsafe or impossible."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(logfname, flags, 0o600)
    except OSError:
        return None
    try:
        if hasattr(os, "getuid") and os.fstat(fd).st_uid != os.getuid():
            os.close(fd)
            return None
        return os.fdopen(fd, "a", errors="backslashreplace")
    except OSError:
        os.close(fd)
        return None


def setup_logging(logfname=None):
    """Log WARNING to stderr, or DEBUG to logfname if given and it can be opened safely."""
    log = logging.getLogger()
    if log.handlers:
        return log  # as with logging.basicConfig, keep a configuration the caller already set up
    stream = None if logfname is None else _open_log_file(logfname)
    if stream is None:
        handler = logging.StreamHandler(sys.stderr)
        level = logging.WARNING
    else:
        handler = logging.StreamHandler(stream)
        level = logging.DEBUG
    handler.setFormatter(EscapingFormatter(LOG_FORMAT))
    log.addHandler(handler)
    log.setLevel(level)
    if logfname is not None and stream is None:
        log.warning(f"not logging to '{logfname}': can't open it, or it is a symlink or "
                    "not owned by you; logging warnings to stderr")
    return log
