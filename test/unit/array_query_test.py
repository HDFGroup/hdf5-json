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
import numpy as np

from h5json.query_util import arrayQuery
from h5json import selections


def _get_test_tabular_data():
    value = [
        ("EBAY", "20170102", 3023, 3088),  # 0
        ("AAPL", "20170102", 3054, 2933),  # 1
        ("AMZN", "20170102", 2973, 3011),  # 2
        ("EBAY", "20170103", 3042, 3128),  # 3
        ("AAPL", "20170103", 3182, 3034),  # 4
        ("AMZN", "20170103", 3021, 2788),  # 5
        ("EBAY", "20170104", 2798, 2876),  # 6
        ("AAPL", "20170104", 2834, 2867),  # 7
        ("AMZN", "20170104", 2891, 2978),  # 8
        ("EBAY", "20170105", 2973, 2962),  # 9
        ("AAPL", "20170105", 2934, 3010),  # 10
        ("AMZN", "20170105", 3018, 3086),  # 11
    ]

    num_rows = len(value)
    shape = (num_rows,)
    dtype = np.dtype([("symbol", "S4"), ("date", "S8"), ("open", "i4"), ("close", "i4")])
    arr = np.zeros(shape, dtype=dtype)
    for i in range(num_rows):
        row = value[i]
        e = arr[i]
        for j in range(4):
            e[j] = row[j]
    return arr


class QueryUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(QueryUtilTest, self).__init__(*args, **kwargs)
        # main
        logging.getLogger().setLevel(logging.ERROR)

    def testArrayQueryNoneCompound(self):
        nrows = 10
        ncols = 10
        shape = (nrows, ncols)
        data_arr = np.zeros(shape, np.float32)
        expected = []
        for i in range(nrows):
            y = i / (nrows - 1.0)
            for j in range(ncols):
                x = j / (ncols - 1.0)
                z = x * x + y * y
                data_arr[i, j] = z
                if z > 1.0:
                    expected.append((i, j))

        query = "field('_') > 1.0"
        result = arrayQuery(query, data_arr)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.dtype, np.dtype("int64"))
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], len(expected))
        self.assertEqual(result.shape[1], 2)
        for i in range(len(expected)):
            e = (result[i][0], result[i][1])
            self.assertTrue(e in expected)

    def testArrayQuery1D(self):

        data_arr = _get_test_tabular_data()
        query = "field('symbol') == b'AAPL'"
        result = arrayQuery(query, data_arr)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.dtype, np.dtype("int64"))
        self.assertEqual(result.shape, (4, 1))
        expected_indexes = (1, 4, 7, 10)  # rows above with AAPL as symbol
        for i in range(4):
            item = result[i]
            self.assertTrue(item[0] in expected_indexes)

        # read just one row back
        result = arrayQuery(query, data_arr, limit=1)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result), 1)
        index = result[0][0]
        self.assertEqual(index, 1)

        # query with selection, no limit
        sel = selections.select(data_arr.shape, (slice(2, 12)))
        result = arrayQuery(query, data_arr, selection=sel)

        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result), 3)
        expected_indexes = (4, 7, 10)
        for i in range(3):
            index = result[i]
            self.assertEqual(index, expected_indexes[i])

        # query with selection and limit
        result = arrayQuery(query, data_arr, limit=2, selection=sel)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result), 2)
        for i in range(2):
            index = result[i]
            self.assertEqual(index, expected_indexes[i])

        # query for row that doesn't exist
        query = "field('symbol') == b'XYZ'"
        result = arrayQuery(query, data_arr)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result), 0)

        # query with isin
        query = "field('open').isin(2798, 2934, 1234)"
        result = arrayQuery(query, data_arr)
        expected_indexes = (6, 10)
        for i in range(2):
            index = result[i]
            self.assertTrue(index in expected_indexes)

        # query with where
        query = "field('symbol').isin(b'AAPL', b'EBAY')"
        result = arrayQuery(query, data_arr)
        expected_indexes = (0, 1, 3, 4, 6, 7, 9, 10)

        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result), 8)
        for i in range(len(result)):
            index = result[i]
            self.assertTrue(index in expected_indexes)

        # boolean query
        query = "(field('symbol').isin(b'AAPL')) & (field('date') > 20170102)"
        result = arrayQuery(query, data_arr)
        self.assertTrue(isinstance(result, np.ndarray))
        expected_indexes = (4, 7, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i]
            self.assertTrue(index in expected_indexes)

        # query with not (~) and isin
        query = "~(field('symbol').isin(b'AAPL', b'EBAY'))"
        result = arrayQuery(query, data_arr)
        expected_indexes = (2, 5, 8, 11)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i]
            self.assertTrue(index in expected_indexes)

        # try bad Limit
        query = "field('symbol') == b'AAPL'"
        try:
            arrayQuery(query, data_arr, limit="foobar")
            self.assertTrue(False)
        except TypeError:
            pass  # expected

        # try invalid query string
        query = "foobar"
        try:
            arrayQuery(query, data_arr)
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # try missing paren
        query = "(field('open') > 5"
        try:
            arrayQuery(query, data_arr)
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # try invalid character
        query = "field('open') @ 5"

        try:
            arrayQuery(query, data_arr)
            self.assertTrue(False)
        except ValueError:
            pass  # expected

    def testArrayQueryEqAlias(self):
        data_arr = _get_test_tabular_data()
        query = "field('symbol') = b'AAPL'"
        result = arrayQuery(query, data_arr)
        expected_indexes = (1, 4, 7, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i]
            self.assertTrue(index in expected_indexes)

    def testArrayQueryIsNullIsValid(self):
        data_arr = np.array([1.0, float("nan"), 3.0, float("nan"), 5.0])

        result = arrayQuery("field('_').is_null()", data_arr)
        expected_indexes = (1, 3)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        result = arrayQuery("field('_').is_valid()", data_arr)
        expected_indexes = (0, 2, 4)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

    def testArrayQueryBareFieldName(self):
        # field("name") and a bare name/'name' are equivalent
        data_arr = _get_test_tabular_data()

        query = "symbol == b'AAPL'"
        result = arrayQuery(query, data_arr)
        expected_indexes = (1, 4, 7, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        query = "symbol.isin(b'AAPL', b'EBAY')"
        result = arrayQuery(query, data_arr)
        expected_indexes = (0, 1, 3, 4, 6, 7, 9, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # bare and field(...) forms can be freely mixed in the same query
        query = "(symbol.isin(b'AAPL')) & (field('date') > 20170102)"
        result = arrayQuery(query, data_arr)
        expected_indexes = (4, 7, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # bare '_' works the same as field('_') for non-compound dtypes
        data_arr_2 = np.array([1.0, 2.5, 3.0, -1.0])
        np.testing.assert_array_equal(
            arrayQuery("_ > 1.0", data_arr_2),
            arrayQuery("field('_') > 1.0", data_arr_2),
        )

    def testArrayQueryWordOperatorSynonyms(self):
        # 'AND'/'OR'/'NOT'/'IN' (case-insensitive) are accepted as word synonyms
        # for '&'/'|'/'~'/'.isin(...)' and can be freely mixed with symbol forms
        data_arr = _get_test_tabular_data()

        # IN as a synonym for .isin(...)
        query = "symbol IN (b'AAPL', b'EBAY')"
        result = arrayQuery(query, data_arr)
        expected_indexes = (0, 1, 3, 4, 6, 7, 9, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # NOT IN
        query = "symbol NOT IN (b'AAPL', b'EBAY')"
        result = arrayQuery(query, data_arr)
        expected_indexes = (2, 5, 8, 11)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # AND
        query = "(symbol IN (b'AAPL')) AND (date > 20170102)"
        result = arrayQuery(query, data_arr)
        expected_indexes = (4, 7, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # OR
        query = "(symbol == b'EBAY') OR (symbol == b'AMZN')"
        result = arrayQuery(query, data_arr)
        expected_indexes = (0, 2, 3, 5, 6, 8, 9, 11)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # NOT
        query = "NOT (symbol == b'AAPL')"
        result = arrayQuery(query, data_arr)
        expected_indexes = (0, 2, 3, 5, 6, 8, 9, 11)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # word and symbol forms are equivalent, case-insensitive, and can be mixed
        np.testing.assert_array_equal(
            arrayQuery("(symbol IN (b'AAPL')) AND (date > 20170102)", data_arr),
            arrayQuery("(symbol.isin(b'AAPL')) & (date > 20170102)", data_arr),
        )
        np.testing.assert_array_equal(
            arrayQuery("symbol IN (b'AAPL') and date > 20170102", data_arr),
            arrayQuery("(symbol IN (b'AAPL')) AND (date > 20170102)", data_arr),
        )

    def testArrayQueryBareValue(self):
        # a bare, unquoted word is treated as a string literal, equivalent to
        # a quoted string or bytes literal - lets fixed-string comparisons skip
        # the b'...'/'...' quoting
        data_arr = _get_test_tabular_data()

        query = "symbol == AAPL"
        result = arrayQuery(query, data_arr)
        expected_indexes = (1, 4, 7, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # bare value works with isin(...) and the IN/NOT IN word synonyms too
        query = "symbol.isin(AAPL, EBAY)"
        result = arrayQuery(query, data_arr)
        expected_indexes = (0, 1, 3, 4, 6, 7, 9, 10)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        query = "symbol IN (AAPL, EBAY)"
        result = arrayQuery(query, data_arr)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        query = "symbol NOT IN (AAPL, EBAY)"
        result = arrayQuery(query, data_arr)
        expected_indexes = (2, 5, 8, 11)
        self.assertEqual(len(result), len(expected_indexes))
        for i in range(len(result)):
            index = result[i][0]
            self.assertTrue(index in expected_indexes)

        # bare and quoted/byte-literal values are equivalent
        np.testing.assert_array_equal(
            arrayQuery("symbol == AAPL", data_arr),
            arrayQuery("symbol == b'AAPL'", data_arr),
        )
        np.testing.assert_array_equal(
            arrayQuery("symbol.isin(AAPL, EBAY)", data_arr),
            arrayQuery("symbol.isin(b'AAPL', b'EBAY')", data_arr),
        )

        # also works for unicode (U-kind) fixed-string dtypes
        dtype_u = np.dtype([("symbol", "U10")])
        data_arr_u = np.array([("AAPL",), ("EBAY",)], dtype=dtype_u)
        result = arrayQuery("symbol == AAPL", data_arr_u)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 0)

    def testArrayQueryFieldCallForCollidingName(self):
        # a field literally named 'field', 'true', 'false', 'and', 'or', 'not' or 'in'
        # can't be written as a bare token (it would tokenize as a keyword) -
        # field(...) is required to disambiguate
        dtype = np.dtype([("field", "i4"), ("true", "i4"), ("and", "i4"), ("in", "i4")])
        data_arr = np.array([(1, 2, 5, 7), (3, 4, 6, 8)], dtype=dtype)

        result = arrayQuery("field('field') > 1", data_arr)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)

        result = arrayQuery("(field('and') > 5) & (field('in') > 7)", data_arr)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)

        try:
            arrayQuery("field > 1", data_arr)
            self.assertTrue(False)
        except ValueError:
            pass  # expected - 'field' bare is parsed as the field(...) keyword

    def testArrayQuery2D(self):

        data_arr = _get_test_tabular_data()

        num_rows = data_arr.shape[0]
        data_arr = data_arr.reshape((int(num_rows / 2), 2))

        query = "field('symbol') == b'AAPL'"
        result = arrayQuery(query, data_arr)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result.dtype), 0)
        expected_indexes = ((0, 1), (2, 0), (3, 1), (5, 0))  # indices with AAPL as symbol
        expected_count = len(expected_indexes)
        self.assertEqual(result.shape, (expected_count, 2))
        for i in range(expected_count):
            item = tuple(result[i])
            self.assertEqual(len(item), 2)  # row and col indexes
            self.assertTrue(item in expected_indexes)

        # read just one row back
        result = arrayQuery(query, data_arr, limit=1)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 2))
        item = result[0]
        self.assertEqual(len(item), 2)
        self.assertTrue(np.array_equal(item, (0, 1)))

        # query with selection
        slices = (slice(0, 6, 1), slice(1, 2, 1))
        sel = selections.select(data_arr.shape, slices)  # select second column
        result = arrayQuery(query, data_arr, selection=sel)
        self.assertTrue(isinstance(result, np.ndarray))
        expected_indexes = ((0, 1), (3, 1))
        expected_count = len(expected_indexes)
        self.assertEqual(result.shape, (expected_count, 2))
        for i in range(expected_count):
            item = tuple(result[i])
            self.assertEqual(item, expected_indexes[i])


if __name__ == "__main__":

    unittest.main()
