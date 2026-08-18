# time_util

Small collection of timestamp helpers: converting a Unix timestamp to an ISO-8601 UTC string, formatting
an elapsed-time duration, and getting the current time with platform-appropriate precision.

## unixTimeToUTC(timestamp)

Converts a Unix timestamp (seconds since 1970-01-01) to an ISO-8601 UTC string. Uses
`datetime.fromtimestamp(timestamp, pytz.utc)` and rewrites the `+00:00` UTC offset suffix that
`isoformat()` produces as `Z` (e.g. `2014-10-30T04:25:21Z`).

## elapsedTime(timestamp)

Returns a human-readable string describing the time elapsed between `timestamp` and the current time
(`time.time()`), broken into days/hours/minutes/seconds — each unit is included only once the elapsed
time is large enough to warrant it (and cascades: once days are shown, hours and minutes are shown too,
even if zero). Returns the literal string `"Invalid timestamp!"` if `timestamp` is in the future.

## getNow(app=None)

Returns the current time as a Unix timestamp. On POSIX systems, returns `time.time()` directly. On
Windows (`os.name == "nt"`), attempts to improve on `time.time()`'s lower clock resolution by deriving
the timestamp from `time.perf_counter()` relative to `app["start_time_relative"]` and `app["start_time"]`,
falling back to `time.time()` if `app` doesn't provide those keys. Raises `ValueError` for any other
`os.name` value.
