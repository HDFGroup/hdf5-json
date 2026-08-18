# track_util

Single-function module for reading the `trackTimes` creation property from a group/dataset/datatype
JSON object (or a bare creation-property-list dict).

## getTrackTimes(obj_json)

Returns the boolean value of `trackTimes` from `obj_json`'s creation properties, or `None` if not set.
`obj_json` may be a full object JSON dict containing a `"creationProperties"` key, or the creation
property list dict itself. `trackTimes` controls whether an HDF5 object records its creation/modification
timestamps.
