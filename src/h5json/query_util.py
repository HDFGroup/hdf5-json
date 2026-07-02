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


import numpy as np


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_KEYWORDS = frozenset({'AND', 'OR', 'NOT', 'XOR', 'IN'})


def _tokenize(query):
    """Tokenize a query string into a list of (type, value) tuples.

    Token types: IDENT, NUMBER, BYTES, STR,
                 EQ NE LT GT LE GE, LPAREN RPAREN COMMA,
                 AND OR NOT XOR IN, EOF.

    Raises ValueError on any character that is not part of the grammar.
    """
    tokens = []
    i = 0
    n = len(query)

    while i < n:
        c = query[i]

        if c.isspace():
            i += 1
            continue

        # Bytes literal:  b'...'  or  b"..."
        if c == 'b' and i + 1 < n and query[i + 1] in ("'", '"'):
            quote = query[i + 1]
            j = i + 2
            while j < n and query[j] != quote:
                j += 1
            if j >= n:
                raise ValueError(f"Unterminated bytes literal at position {i}")
            tokens.append(('BYTES', query[i + 2:j].encode('latin-1')))
            i = j + 1
            continue

        # String literal:  '...'  or  "..."
        if c in ("'", '"'):
            quote = c
            j = i + 1
            while j < n and query[j] != quote:
                j += 1
            if j >= n:
                raise ValueError(f"Unterminated string literal at position {i}")
            tokens.append(('STR', query[i + 1:j]))
            i = j + 1
            continue

        # Identifiers and keywords (case-insensitive for keywords)
        if c.isalpha() or c == '_':
            j = i
            while j < n and (query[j].isalnum() or query[j] == '_'):
                j += 1
            word = query[i:j]
            upper = word.upper()
            tokens.append((upper if upper in _KEYWORDS else 'IDENT', word if upper not in _KEYWORDS else None))
            i = j
            continue

        # Numbers: optional leading '-', digits, optional '.' + digits
        if c.isdigit() or (c == '-' and i + 1 < n and query[i + 1].isdigit()):
            j = i + (1 if c == '-' else 0)
            while j < n and query[j].isdigit():
                j += 1
            is_float = j < n and query[j] == '.'
            if is_float:
                j += 1
                while j < n and query[j].isdigit():
                    j += 1
            val = float(query[i:j]) if is_float else int(query[i:j])
            tokens.append(('NUMBER', val))
            i = j
            continue

        # Two-character operators (checked before single-char)
        two = query[i:i + 2]
        if two == '==':
            tokens.append(('EQ', None))
            i += 2
            continue
        if two == '!=':
            tokens.append(('NE', None))
            i += 2
            continue
        if two == '<=':
            tokens.append(('LE', None))
            i += 2
            continue
        if two == '>=':
            tokens.append(('GE', None))
            i += 2
            continue

        # Single-character operators and punctuation
        if c == '<':
            tokens.append(('LT', None))
            i += 1
            continue
        if c == '>':
            tokens.append(('GT', None))
            i += 1
            continue
        if c == '(':
            tokens.append(('LPAREN', None))
            i += 1
            continue
        if c == ')':
            tokens.append(('RPAREN', None))
            i += 1
            continue
        if c == ',':
            tokens.append(('COMMA', None))
            i += 1
            continue

        raise ValueError(f"Invalid character {c!r} at position {i} in query: {query!r}")

    tokens.append(('EOF', None))
    return tokens


# ---------------------------------------------------------------------------
# Parser  (recursive descent)
# ---------------------------------------------------------------------------
# AST nodes are plain tuples:
#   ('CMP',    field, op, value)       op in EQ NE LT GT LE GE
#   ('IN',     field, values)
#   ('NOT_IN', field, values)
#   ('AND',    left, right)
#   ('OR',     left, right)
#   ('XOR',    left, right)
#   ('NOT',    sub)
#
# field is None for the '_' wildcard (non-compound dtypes), otherwise a str.

class _Parser:
    def __init__(self, tokens, dtype):
        self._tokens = tokens
        self._pos = 0
        self._field_names = set(dtype.names) if dtype.names else None

    def _peek(self):
        return self._tokens[self._pos]

    def _consume(self, expected=None):
        tok = self._tokens[self._pos]
        if expected is not None and tok[0] != expected:
            raise ValueError(f"Expected {expected!r} but got {tok[0]!r} (token {self._pos})")
        self._pos += 1
        return tok

    def parse(self):
        expr = self._parse_or()
        if self._peek()[0] != 'EOF':
            raise ValueError(f"Unexpected token {self._peek()!r} after expression")
        return expr

    def _parse_or(self):
        left = self._parse_xor()
        while self._peek()[0] == 'OR':
            self._consume()
            left = ('OR', left, self._parse_xor())
        return left

    def _parse_xor(self):
        left = self._parse_and()
        while self._peek()[0] == 'XOR':
            self._consume()
            left = ('XOR', left, self._parse_and())
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._peek()[0] == 'AND':
            self._consume()
            left = ('AND', left, self._parse_not())
        return left

    def _parse_not(self):
        if self._peek()[0] == 'NOT':
            self._consume()
            return ('NOT', self._parse_not())
        return self._parse_primary()

    def _parse_primary(self):
        tok = self._peek()

        if tok[0] == 'LPAREN':
            self._consume()
            expr = self._parse_or()
            if self._peek()[0] != 'RPAREN':
                raise ValueError("Missing closing ')' in query")
            self._consume()
            return expr

        # Field name: bare identifier or single-quoted name like 'date'
        if tok[0] in ('IDENT', 'STR'):
            field = tok[1]
            self._consume()

            if self._field_names is None:
                # Non-compound dtype: only '_' is allowed
                if field != '_':
                    raise ValueError(
                        f"Field {field!r} is not valid for non-compound dtype; use '_'"
                    )
                field = None  # None means "use the array element itself"
            else:
                if field not in self._field_names:
                    raise ValueError(
                        f"Field {field!r} not found in dtype "
                        f"(available: {sorted(self._field_names)})"
                    )

            next_type = self._peek()[0]

            # FIELD NOT IN (...)
            if next_type == 'NOT':
                self._consume()
                if self._peek()[0] != 'IN':
                    raise ValueError("Expected 'IN' after 'NOT'")
                self._consume()
                return ('NOT_IN', field, self._parse_value_list())

            # FIELD IN (...)
            if next_type == 'IN':
                self._consume()
                return ('IN', field, self._parse_value_list())

            # FIELD op VALUE
            if next_type not in ('EQ', 'NE', 'LT', 'GT', 'LE', 'GE'):
                raise ValueError(
                    f"Expected a comparison operator after field, got {next_type!r}"
                )
            self._consume()
            return ('CMP', field, next_type, self._parse_value())

        raise ValueError(
            f"Unexpected token {tok!r} — expected a field name or '('"
        )

    def _parse_value_list(self):
        if self._peek()[0] != 'LPAREN':
            raise ValueError("Expected '(' to open IN value list")
        self._consume()
        values = [self._parse_value()]
        while self._peek()[0] == 'COMMA':
            self._consume()
            values.append(self._parse_value())
        if self._peek()[0] != 'RPAREN':
            raise ValueError("Expected ')' to close IN value list")
        self._consume()
        return tuple(values)

    def _parse_value(self):
        tok = self._peek()
        if tok[0] in ('NUMBER', 'BYTES', 'STR'):
            self._consume()
            return tok[1]
        raise ValueError(f"Expected a literal value, got {tok[0]!r}")


# ---------------------------------------------------------------------------
# Evaluator  (all operations are vectorised NumPy — no Python loops over rows)
# ---------------------------------------------------------------------------

def _coerce_value(value, field_arr):
    """Coerce a scalar comparison value to match field_arr's dtype.

    Handles the common case where a bytes/string field is compared to a
    number (e.g. a YYYYMMDD date stored as S8 compared to an integer).
    """
    kind = field_arr.dtype.kind
    if kind == 'S':  # fixed-width byte string
        if isinstance(value, (int, float)):
            return str(int(value)).encode('ascii')
        if isinstance(value, str):
            return value.encode('ascii')
    elif kind == 'U':  # fixed-width unicode string
        if isinstance(value, (int, float)):
            return str(int(value))
        if isinstance(value, bytes):
            return value.decode('ascii')
    return value


def _evaluate(expr, arr):
    """Evaluate an AST expression against arr, returning a boolean mask."""
    kind = expr[0]

    if kind == 'CMP':
        _, field, op, value = expr
        field_arr = arr if field is None else arr[field]
        value = _coerce_value(value, field_arr)
        if op == 'EQ': return field_arr == value  # noqa: E701
        if op == 'NE': return field_arr != value  # noqa: E701
        if op == 'LT': return field_arr < value   # noqa: E701
        if op == 'GT': return field_arr > value   # noqa: E701
        if op == 'LE': return field_arr <= value  # noqa: E701
        if op == 'GE': return field_arr >= value  # noqa: E701

    if kind in ('IN', 'NOT_IN'):
        _, field, values = expr
        field_arr = arr if field is None else arr[field]
        coerced = [_coerce_value(v, field_arr) for v in values]
        mask = np.isin(field_arr, coerced)
        return ~mask if kind == 'NOT_IN' else mask

    if kind == 'AND':
        return _evaluate(expr[1], arr) & _evaluate(expr[2], arr)
    if kind == 'OR':
        return _evaluate(expr[1], arr) | _evaluate(expr[2], arr)
    if kind == 'XOR':
        return _evaluate(expr[1], arr) ^ _evaluate(expr[2], arr)
    if kind == 'NOT':
        return ~_evaluate(expr[1], arr)

    raise ValueError(f"Unknown AST node: {kind!r}")


# ---------------------------------------------------------------------------
# Selection helper
# ---------------------------------------------------------------------------

def _selection_to_mask(selection, shape):
    """Return a boolean NumPy mask of *shape* with True for every selected element."""
    mask = np.zeros(shape, dtype=bool)
    mask[selection.slices] = True
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def arrayQuery(
    query,
    data_arr,
    selection=None,
    limit=0,
):
    """
    Return an ndarray of indexes of the given data_arr where the data_arr element satisfy the query condition.

    query: A sql-like query string.  If data_arr type is a simple dtype, the only variable allowed in
      the query string is '_'.  If data_arr is a compond dtype, a variable can be any sub-type name of the dtype.
      variables can be compared using the following operators:
          '==': the value is equal to the array element (or element sub-field)
          '!=': the value is not equal
          '<': the value is less than
          '>': the value is greater than
          '<=': the value is less than or equal
          '>=': the value is greater than or equal
          'IN': the value is in the given set
          Multiple varibles and/or conditions can be combined using the boolean opeators 'NOT', 'AND', 'OR', 'XOR'

    if selection is not None, only elements within the given selection are considered

    if limit is not 0, only up to limit indices will be returned

    The return value will be an ndarray.  The array shape (count, rank) where rank is the number of array dimensions.

    Example queries:
        "_ > 1.0" # match any array element with a value greater than 1.0
        "symbol == b'AAPL'"  # match any array element where the symbol field is b'AAPL' (for ascii numpy string dtypes)
        "symbol == 'AAPL'"  # match any array element where the symbol field is 'AAPL' (for unicode numpy string dtypes)
        "symbol IN ('AAPL', 'EBAY')"   # match any array element where the symbol field is 'AAPL' or 'EBAY'
        "symbol IN ('AAPL', 'EBAY') AND 'date' > 20170102"
        "symbol NOT IN ('AAPL', 'EBAY') AND 'date' > 20170102"

    """
    if not isinstance(data_arr, np.ndarray):
        raise TypeError("unexpected array type")

    if limit != 0 and not isinstance(limit, int):
        raise TypeError("limit must be an integer")

    dims = data_arr.shape
    rank = len(dims)

    if rank == 0:
        raise ValueError("query is not supported for scalar arrays")

    # Parse (also validates characters and field names — no eval() used)
    tokens = _tokenize(query)
    expr = _Parser(tokens, data_arr.dtype).parse()

    # Evaluate: entirely vectorised NumPy operations, no per-row Python loops
    mask = _evaluate(expr, data_arr)

    if selection is not None:
        mask = mask & _selection_to_mask(selection, dims)

    indices = np.argwhere(mask)
    if limit > 0:
        indices = indices[:limit]
    return indices
