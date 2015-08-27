from cStringIO import StringIO


class StringStore(object):
    """Store and concatenate strings fast."""

    def __init__(self):
        # Container for storing strings...
        self._s = StringIO()

    def append(self, string):
        """Append the new ``string`` to the rest of collection.

        :arg str string: New string.
        """
        self._s.write(string)

    def dump(self):
        """Dump all the stored strings as one."""
        return self._s.getvalue()
