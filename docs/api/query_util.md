# query_util

Expression parser and vectorised evaluator for querying dataset values, used by
`Hdf5db.getDatasetValues` and `Hdf5db.queryDataset` to select array elements/rows matching a boolean
condition. The module implements a small tokenizer, a recursive-descent parser producing a tuple-based
AST, and a NumPy-vectorised evaluator — no `eval()` is used, and no per-row Python loops are involved.
Only one function, `arrayQuery`, is part of the public API; the tokenizer, parser (`_Parser`), and
evaluator internals are private.

The query syntax is compatible with [h5col](https://hdfgroup.github.io/h5col/queries/syntax.html),
extended so that `AND`/`OR`/`NOT`/`IN` may be used as case-insensitive word synonyms for `&`/`|`/`~`/
`.isin(...)`. Fields are referenced as `field("name")` or a bare name/`'name'`; for non-compound dtypes
the array element itself is referenced as `field("_")` or `_`. Supported predicates are equality/
comparison operators (`==`, `=`, `!=`, `<`, `<=`, `>`, `>=`), `.isin(...)`/`IN (...)`/`NOT IN (...)`, and
`.is_null()`/`.is_valid()`; predicates combine with `&`/`AND`, `|`/`OR`, `~`/`NOT`, with parentheses for
grouping.

## arrayQuery(query, data_arr, selection=None, limit=0)

Parses `query` and evaluates it against `data_arr`, returning an `ndarray` of indices (shape
`(count, rank)`, from `numpy.argwhere`) where the corresponding array elements satisfy the condition.

- `query` — a query string in the syntax described above.
- `data_arr` — a NumPy `ndarray`; must not be scalar (rank 0 raises `ValueError`).
- `selection` — optional selection object (from `selections.py`); if given, results are additionally
  masked to `selection.slices` via `_selection_to_mask`, restricting matches to that region.
- `limit` — if nonzero, truncates the returned indices to at most `limit` rows.

Field names are validated against `data_arr.dtype.names` (for compound dtypes) or restricted to `_` for
simple dtypes, and comparison values are coerced to match the target field's dtype (e.g. numbers are
converted to fixed-width byte/unicode strings when compared against `S`/`U` fields, so `symbol == AAPL`
and `symbol == b'AAPL'` behave the same way). `is_null()`/`is_valid()` only recognize missing values for
float (`NaN`) and object dtypes; other dtypes are treated as never-null. Raises `TypeError` if `data_arr`
is not an `ndarray` or `limit` is not an `int`, and `ValueError` for any syntax error, unknown field, or
malformed expression.
