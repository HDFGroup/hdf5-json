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

_KEYWORDS = frozenset({'FIELD', 'TRUE', 'FALSE', 'AND', 'OR', 'NOT', 'IN'})


def _tokenize(query):
    """Tokenize a query string into a list of (type, value) tuples.

    Token types: FIELD, IDENT, NUMBER, BYTES, STR, BOOL,
                 EQ NE LT GT LE GE, LPAREN RPAREN COMMA DOT,
                 AMP PIPE TILDE, AND OR NOT IN, EOF.

    'AND'/'OR'/'NOT'/'IN' are accepted as case-insensitive word synonyms for
    '&'/'|'/'~'/'.isin(...)' respectively — either spelling may be used.

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
            if upper == 'TRUE':
                tokens.append(('BOOL', True))
            elif upper == 'FALSE':
                tokens.append(('BOOL', False))
            elif upper == 'FIELD':
                tokens.append(('FIELD', None))
            elif upper in ('AND', 'OR', 'NOT', 'IN'):
                tokens.append((upper, None))
            else:
                tokens.append(('IDENT', word))
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
        if c == '=':
            tokens.append(('EQ', None))
            i += 1
            continue
        if c == '<':
            tokens.append(('LT', None))
            i += 1
            continue
        if c == '>':
            tokens.append(('GT', None))
            i += 1
            continue
        if c == '&':
            tokens.append(('AMP', None))
            i += 1
            continue
        if c == '|':
            tokens.append(('PIPE', None))
            i += 1
            continue
        if c == '~':
            tokens.append(('TILDE', None))
            i += 1
            continue
        if c == '.':
            tokens.append(('DOT', None))
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
#   ('CMP',      field, op, value)     op in EQ NE LT GT LE GE
#   ('IN',       field, values)
#   ('NOT_IN',   field, values)
#   ('IS_NULL',  field)
#   ('IS_VALID', field)
#   ('AND',      left, right)
#   ('OR',       left, right)
#   ('NOT',      sub)
#
# field is None for the '_' wildcard (non-compound dtypes), otherwise a str.
#
# Grammar (h5col-compatible, with AND/OR/NOT/IN accepted as word synonyms):
#   expr        := or_expr
#   or_expr     := and_expr ( ('|' | OR) and_expr )*
#   and_expr    := not_expr ( ('&' | AND) not_expr )*
#   not_expr    := ('~' | NOT) not_expr | primary
#   primary     := '(' expr ')' | predicate
#   predicate   := field_name ( cmp_op value
#                             | '.' 'isin' '(' value (',' value)* ')'
#                             | IN '(' value (',' value)* ')'
#                             | NOT IN '(' value (',' value)* ')'
#                             | '.' 'is_null' '(' ')'
#                             | '.' 'is_valid' '(' ')' )
#   field_name  := FIELD '(' (STR | IDENT) ')' | STR | IDENT
#   cmp_op      := '==' | '=' | '!=' | '<' | '<=' | '>' | '>='
#   value       := NUMBER | STR | BYTES | BOOL | IDENT   (bare IDENT == quoted STR)
#
# field("name") and a bare name/'name' are equivalent; field(...) is only
# needed to quote a name that would otherwise be unparsable as a bare token
# (e.g. it collides with 'field'/'true'/'false'/'and'/'or'/'not'/'in', or
# contains punctuation).

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
        left = self._parse_and()
        while self._peek()[0] in ('PIPE', 'OR'):
            self._consume()
            left = ('OR', left, self._parse_and())
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._peek()[0] in ('AMP', 'AND'):
            self._consume()
            left = ('AND', left, self._parse_not())
        return left

    def _parse_not(self):
        if self._peek()[0] in ('TILDE', 'NOT'):
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

        return self._parse_predicate()

    def _parse_field(self):
        tok = self._peek()

        if tok[0] == 'FIELD':
            self._consume()
            self._consume('LPAREN')
            name_tok = self._peek()
            if name_tok[0] not in ('STR', 'IDENT'):
                raise ValueError(f"Expected a field name inside field(...), got {name_tok[0]!r}")
            self._consume()
            name = name_tok[1]
            if self._peek()[0] != 'RPAREN':
                raise ValueError("Expected ')' to close field(...)")
            self._consume()
        elif tok[0] in ('STR', 'IDENT'):
            self._consume()
            name = tok[1]
        else:
            raise ValueError(
                f"Unexpected token {tok!r} — expected a field name or field(...)"
            )

        if self._field_names is None:
            # Non-compound dtype: only '_' is allowed
            if name != '_':
                raise ValueError(
                    f"Field {name!r} is not valid for non-compound dtype; use '_' or field('_')"
                )
            return None  # None means "use the array element itself"

        if name not in self._field_names:
            raise ValueError(
                f"Field {name!r} not found in dtype (available: {sorted(self._field_names)})"
            )
        return name

    def _parse_predicate(self):
        field = self._parse_field()

        # Method call:  field(...).isin(...)  /  .is_null()  /  .is_valid()
        if self._peek()[0] == 'DOT':
            self._consume()
            method_tok = self._consume('IDENT')
            method = method_tok[1].lower()

            if method == 'isin':
                return ('IN', field, self._parse_paren_value_list())

            if method == 'is_null':
                self._consume('LPAREN')
                if self._peek()[0] != 'RPAREN':
                    raise ValueError("is_null() takes no arguments")
                self._consume()
                return ('IS_NULL', field)

            if method == 'is_valid':
                self._consume('LPAREN')
                if self._peek()[0] != 'RPAREN':
                    raise ValueError("is_valid() takes no arguments")
                self._consume()
                return ('IS_VALID', field)

            raise ValueError(f"Unknown method {method_tok[1]!r} — expected 'isin', 'is_null' or 'is_valid'")

        # field IN (...)  /  field NOT IN (...)   (word synonyms for .isin(...))
        if self._peek()[0] == 'IN':
            self._consume()
            return ('IN', field, self._parse_paren_value_list())

        if self._peek()[0] == 'NOT':
            self._consume()
            if self._peek()[0] != 'IN':
                raise ValueError("Expected 'IN' after 'NOT'")
            self._consume()
            return ('NOT_IN', field, self._parse_paren_value_list())

        # field(...) op value
        next_type = self._peek()[0]
        if next_type not in ('EQ', 'NE', 'LT', 'GT', 'LE', 'GE'):
            raise ValueError(
                f"Expected a comparison operator or '.' after field(...), got {next_type!r}"
            )
        self._consume()
        return ('CMP', field, next_type, self._parse_value())

    def _parse_paren_value_list(self):
        self._consume('LPAREN')
        values = [self._parse_value()]
        while self._peek()[0] == 'COMMA':
            self._consume()
            values.append(self._parse_value())
        if self._peek()[0] != 'RPAREN':
            raise ValueError("Expected ')' to close value list")
        self._consume()
        return tuple(values)

    def _parse_value(self):
        tok = self._peek()
        if tok[0] in ('NUMBER', 'BYTES', 'STR', 'BOOL'):
            self._consume()
            return tok[1]
        if tok[0] == 'IDENT':
            # a bare, unquoted word is treated as a string literal, same as 'word' —
            # lets fixed-string comparisons skip the b'...'/'...' quoting, e.g.
            # symbol == AAPL. Values that collide with a keyword (true/false/field/
            # and/or/not/in) still need quotes to be used as a literal string.
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
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return str(int(value)).encode('ascii')
        if isinstance(value, str):
            return value.encode('ascii')
    elif kind == 'U':  # fixed-width unicode string
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return str(int(value))
        if isinstance(value, bytes):
            return value.decode('ascii')
    return value


def _is_null_mask(field_arr):
    """Boolean mask that is True where field_arr holds a missing value.

    Fixed-width numeric/string dtypes have no missing-value representation
    (aside from NaN for floats), so only float and object dtypes can be null.
    """
    kind = field_arr.dtype.kind
    if kind == 'f':
        return np.isnan(field_arr)
    if kind == 'O':
        return field_arr == None  # noqa: E711 (vectorised elementwise via numpy for object dtype)
    return np.zeros(field_arr.shape, dtype=bool)


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

    if kind == 'IS_NULL':
        _, field = expr
        field_arr = arr if field is None else arr[field]
        return _is_null_mask(field_arr)

    if kind == 'IS_VALID':
        _, field = expr
        field_arr = arr if field is None else arr[field]
        return ~_is_null_mask(field_arr)

    if kind == 'AND':
        return _evaluate(expr[1], arr) & _evaluate(expr[2], arr)
    if kind == 'OR':
        return _evaluate(expr[1], arr) | _evaluate(expr[2], arr)
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

    query: a query string compatible with the h5col query syntax
      (https://hdfgroup.github.io/h5col/queries/syntax.html), with 'AND'/'OR'/'NOT'/'IN'
      also accepted as case-insensitive word synonyms. Fields are referenced with
      field("name"), or simply the bare name/'name' — for a non-compound dtype, the array
      element itself is field("_") or just '_'. field(...) is only needed to quote a name
      that can't be written as a bare token (e.g. it collides with a keyword such as
      'field'/'true'/'false'/'and'/'or'/'not'/'in', or contains characters outside
      [A-Za-z0-9_]).
      Predicates:
          field("x") == v   (or '=')     : equal to v
          field("x") != v                 : not equal to v
          field("x") < v                   : less than v
          field("x") <= v                  : less than or equal to v
          field("x") > v                   : greater than v
          field("x") >= v                  : greater than or equal to v
          field("x").isin(v1, v2, ...)     : x is one of the given values
          field("x") IN (v1, v2, ...)      : same as .isin(...)
          field("x") NOT IN (v1, v2, ...)  : x is none of the given values
          field("x").is_null()             : x is a missing value
          field("x").is_valid()            : x is not a missing value
      Predicates combine with the boolean operators '&'/AND, '|'/OR, '~'/NOT;
      parenthesize sub-expressions freely, e.g. (a) & (b) or (a) AND (b).

      A value may also be a bare, unquoted word (e.g. AAPL instead of 'AAPL' or
      b'AAPL') — this is equivalent to a quoted string and is coerced to bytes/str
      to match the field's dtype same as a quoted literal. A value that collides
      with a keyword (true/false/field/and/or/not/in) still needs quotes.

    if selection is not None, only elements within the given selection are considered

    if limit is not 0, only up to limit indices will be returned

    The return value will be an ndarray.  The array shape (count, rank) where rank is the number of array dimensions.

    Example queries:
        "_ > 1.0"    # bare '_' — match any array element with a value greater than 1.0
        "field('_') > 1.0"    # equivalent, using field(...)
        "symbol == b'AAPL'"   # bare field name — match any element where symbol is b'AAPL'
        "symbol == AAPL"      # same, using a bare (unquoted) value instead of b'AAPL'
        "field('symbol') == 'AAPL'"    # match any element where symbol is 'AAPL' (unicode dtype)
        "symbol.isin('AAPL', 'EBAY')"   # match any element where symbol is 'AAPL' or 'EBAY'
        "symbol IN (AAPL, EBAY)"        # same, using bare values and the 'IN' word synonym
        "(symbol.isin('AAPL', 'EBAY')) & (field('date') > 20170102)"   # mixing bare/field(...) forms
        "~(symbol.isin('AAPL', 'EBAY')) & (date > 20170102)"

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
