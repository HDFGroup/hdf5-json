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

from h5json.filters import FILTER_DEFS
from h5json.filters import getFilterItem, validateFilter, isCompressionFilter
from h5json.filters import getAllFilterNames, getCompressionFilter, getShuffleFilter
from h5json.filters import normalizeFilters, validateFilters


class FiltersTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FiltersTest, self).__init__(*args, **kwargs)
        # main
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testStandardFilters(self):

        # check standard filters with no options

        self.assertEqual(len(FILTER_DEFS), 14)
        for item in FILTER_DEFS:
            filter_class = item[0]
            filter_id = item[1]
            filter_name = item[2]
            for value in (filter_class, filter_id, filter_name):
                filter_json = getFilterItem(value)
                validateFilter(filter_json)

        # check alternate names work
        for name in ("deflate", "gzip"):
            filter_json = getFilterItem(name)
            validateFilter(filter_json)
            self.assertTrue(isCompressionFilter(filter_json))

        # check random name raises exception
        try:
            getFilterItem("goofy")
            self.assertTrue(False)
        except KeyError:
            pass  # expected

        # check invalid filter id fails
        try:
            getFilterItem(1234)
            self.assertTrue(False)
        except KeyError:
            pass  # expected

    def testCustomFilters(self):

        # check custom filter usage
        custom_filter = {"class": "H5Z_FILTER_USER", "name": "myspecialfilter"}
        # id should be over 32000
        custom_filter["id"] = 32000
        try:
            validateFilter(custom_filter)
            self.assertTrue(False)  # shouldn't get here
        except ValueError:
            pass  # expected

        custom_filter["id"] = 32099
        validateFilter(custom_filter)

        custom_filter["unknown_option"] = 42
        try:
            validateFilter(custom_filter)
            self.assertTrue(False)  # shouldn't get here
        except KeyError:
            pass  # expected

        del custom_filter["unknown_option"]
        good_params = (1, 2, 3)
        bad_params = (2, -1)  # needs to be positive
        custom_filter["parameters"] = good_params
        validateFilter(custom_filter)

        custom_filter["parameters"] = bad_params
        try:
            validateFilter(custom_filter)
            self.assertTrue(False)  # shouldn't get here
        except TypeError:
            pass  # expected

    def testGetAllFilterNames(self):
        names = getAllFilterNames()
        # tuple of unique, sorted filter names - excludes "none" (id 0)
        self.assertEqual(type(names), tuple)
        self.assertEqual(len(names), 13)
        self.assertEqual(names, tuple(sorted(names)))
        self.assertEqual(len(names), len(set(names)))
        for expected_name in ("gzip", "szip", "shuffle", "fletcher32", "lzf", "zstd"):
            self.assertTrue(expected_name in names)
        self.assertFalse("none" in names)
        # class keys shouldn't show up, only the plain names
        self.assertFalse("H5Z_FILTER_DEFLATE" in names)

    def testGetCompressionAndShuffleFilters(self):
        gzip_filter = getFilterItem("gzip")
        shuffle_filter = getFilterItem("shuffle")
        fletcher_filter = getFilterItem("fletcher32")

        filters = [shuffle_filter, gzip_filter, fletcher_filter]
        compression_filter = getCompressionFilter(filters)
        self.assertEqual(compression_filter, gzip_filter)
        found_shuffle_filter = getShuffleFilter(filters)
        self.assertEqual(found_shuffle_filter, shuffle_filter)

        # no compression filter present
        no_compression = [shuffle_filter, fletcher_filter]
        self.assertEqual(getCompressionFilter(no_compression), None)

        # no shuffle filter present
        no_shuffle = [gzip_filter, fletcher_filter]
        self.assertEqual(getShuffleFilter(no_shuffle), None)

        # empty filter list
        self.assertEqual(getCompressionFilter([]), None)
        self.assertEqual(getShuffleFilter([]), None)

    def testNormalizeFilters(self):
        deflate = {"class": "H5Z_FILTER_DEFLATE", "id": 1, "name": "gzip"}
        lz4 = {"class": "H5Z_FILTER_LZ4", "id": 32004, "name": "lz4"}

        # specs with class and id but no name
        for spec, expected in (
            ({"class": "H5Z_FILTER_DEFLATE", "id": 1, "level": 9}, dict(deflate, level=9)),
            ({"class": "H5Z_FILTER_SHUFFLE", "id": 2},
             {"class": "H5Z_FILTER_SHUFFLE", "id": 2, "name": "shuffle"}),
            ({"class": "H5Z_FILTER_FLETCHER32", "id": 3},
             {"class": "H5Z_FILTER_FLETCHER32", "id": 3, "name": "fletcher32"}),
        ):
            normalized = normalizeFilters([spec])
            self.assertEqual(normalized, [expected])
            validateFilters(normalized)

        self.assertEqual(normalizeFilters([{"class": "H5Z_FILTER_DEFLATE"}]), [deflate])

        # a user filter naming or numbering a registered one gets that filter's
        # class, so its options validate (lz4's "level" isn't a user filter option)
        for spec in ({"class": "H5Z_FILTER_USER", "name": "lz4", "level": 5},
                     {"class": "H5Z_FILTER_USER", "id": 32004, "level": 5}):
            normalized = normalizeFilters([spec])
            self.assertEqual(normalized, [dict(lz4, level=5)])
            validateFilters(normalized)
        # with the deflate/zlib aliases resolving to gzip
        spec = {"class": "H5Z_FILTER_USER", "name": "deflate"}
        self.assertEqual(normalizeFilters([spec])[0]["class"], "H5Z_FILTER_DEFLATE")

        # a name the client gave is kept, order is preserved, and the input isn't
        # modified
        spec = {"class": "H5Z_FILTER_DEFLATE", "name": "deflate"}
        filters = [{"class": "H5Z_FILTER_SHUFFLE"}, spec]
        normalized = normalizeFilters(filters)
        self.assertEqual([f["class"] for f in normalized],
                         ["H5Z_FILTER_SHUFFLE", "H5Z_FILTER_DEFLATE"])
        self.assertEqual(normalized[1]["name"], "deflate")
        self.assertEqual(spec, {"class": "H5Z_FILTER_DEFLATE", "name": "deflate"})

        # a wrong id is kept
        normalized = normalizeFilters([{"class": "H5Z_FILTER_DEFLATE", "id": 2}])
        self.assertEqual(normalized[0]["id"], 2)
        with self.assertRaises(ValueError):
            validateFilters(normalized)

        # anything else passes through unchanged
        # unknown filters, dicts without a class, and bare names or ids
        for spec in ({"class": "H5Z_FILTER_USER", "name": "incorrect_filter"},
                     {"class": "H5Z_FILTER_FOOBAR"},
                     {"id": 1},
                     "gzip",
                     1):
            self.assertEqual(normalizeFilters([spec]), [spec])
            with self.assertRaises((KeyError, TypeError, ValueError)):
                validateFilters(normalizeFilters([spec]))

        # an unregistered user filter with a complete spec is left alone
        spec = {"class": "H5Z_FILTER_USER", "id": 40000, "name": "custom"}
        self.assertEqual(normalizeFilters([spec]), [spec])

        # a non-list is returned unchanged
        self.assertEqual(normalizeFilters(None), None)


if __name__ == "__main__":
    # setup test files

    unittest.main()
