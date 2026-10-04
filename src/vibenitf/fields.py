"""Primitives for reading NITF's fixed-length ASCII field encoding.

Every NITF header field occupies a fixed number of bytes. Textual fields are
space padded on the right, numeric fields are normally zero filled on the
left, and a handful are raw binary. Fields are never delimited and never
self-describing, so a header can only be read by consuming its fields in
order -- which is what `Cursor` does.
"""

from .errors import NitfParseError, NitfTruncatedError


class Cursor:
    """A forward-only cursor over a header's bytes.

    The cursor records the name and raw text of every field it consumes in
    `raw`, so callers can recover values this package does not interpret, and
    so error messages can point at a real field name.
    """

    __slots__ = ("buf", "pos", "base", "raw", "_strict")

    def __init__(self, buf, start=0, base=0, strict=True):
        self.buf = buf
        self.pos = start
        self.base = base
        self.raw = {}
        self._strict = strict

    # -- low level ---------------------------------------------------------

    @property
    def offset(self):
        """Absolute offset in the file of the next unread byte."""
        return self.base + self.pos

    @property
    def remaining(self):
        return len(self.buf) - self.pos

    def take(self, length, name=None):
        """Consume `length` raw bytes."""
        if length < 0:
            raise ValueError("negative field length for %s" % name)
        end = self.pos + length
        if end > len(self.buf):
            raise NitfTruncatedError(
                "need %d bytes but only %d remain" % (length, self.remaining),
                offset=self.offset,
                field=name,
            )
        chunk = self.buf[self.pos : end]
        self.pos = end
        return chunk

    def skip(self, length, name=None):
        self.take(length, name)

    # -- typed readers -----------------------------------------------------

    def bytes(self, length, name):
        """A binary field, recorded as a hex string."""
        chunk = self.take(length, name)
        self.raw[name] = chunk.hex()
        return chunk

    def text(self, length, name):
        """A BCS-A text field, returned with trailing padding removed.

        NITF text is restricted to the ASCII subset BCS-A, but real files do
        contain stray bytes, so decoding is latin-1 (which cannot fail) rather
        than ascii. The raw field text is kept verbatim in `raw`.
        """
        chunk = self.take(length, name)
        value = chunk.decode("latin-1")
        self.raw[name] = value
        return value.rstrip(" \t\r\n\0")

    def int(self, length, name, default=None, allow_blank=False):
        """A numeric field.

        Blank numeric fields are illegal but common in the wild. When
        `allow_blank` is set, or the cursor is not strict, a blank field
        yields `default` instead of raising.
        """
        chunk = self.take(length, name)
        value = chunk.decode("latin-1")
        self.raw[name] = value
        text = value.strip()
        if not text:
            if allow_blank or not self._strict:
                return default
            raise NitfParseError(
                "blank where a number is required", offset=self.offset - length, field=name
            )
        try:
            return int(text, 10)
        except ValueError:
            if not self._strict:
                return default
            raise NitfParseError(
                "%r is not a decimal number" % value,
                offset=self.offset - length,
                field=name,
            ) from None

    def float(self, length, name, default=None):
        """A real-valued field, as used by several TREs."""
        chunk = self.take(length, name)
        value = chunk.decode("latin-1")
        self.raw[name] = value
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            if not self._strict:
                return default
            raise NitfParseError(
                "%r is not a real number" % value,
                offset=self.offset - length,
                field=name,
            ) from None

    def indexed(self, length, name, index, reader="text", **kwargs):
        """Read a field belonging to a repeating group.

        NITF names repeated fields with a 1-based ordinal suffix, e.g. the
        third image comment is ICOM3. This keeps `raw` faithful to those
        names.
        """
        return getattr(self, reader)(length, "%s%d" % (name, index), **kwargs)
