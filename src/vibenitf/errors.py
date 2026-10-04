"""Exception hierarchy for vibenitf."""


class NitfError(Exception):
    """Base class for every error raised by this package."""


class NitfParseError(NitfError):
    """The byte stream does not conform to the NITF specification."""

    def __init__(self, message, offset=None, field=None):
        self.offset = offset
        self.field = field
        where = []
        if field:
            where.append("field %s" % field)
        if offset is not None:
            where.append("offset %d" % offset)
        if where:
            message = "%s (%s)" % (message, ", ".join(where))
        super().__init__(message)


class NitfTruncatedError(NitfParseError):
    """The file ended before a declared structure was complete."""


class UnsupportedCompressionError(NitfError):
    """The image uses a compression scheme this build cannot decode.

    ``code`` is the two-character IC value; ``reason`` explains whether the
    scheme is unimplemented outright or needs an optional dependency.
    """

    def __init__(self, code, reason):
        self.code = code
        self.reason = reason
        super().__init__("IC=%s: %s" % (code, reason))


class UnsupportedFeatureError(NitfError):
    """A legal NITF construct that this reader does not implement."""
