"""Tagged record extensions.

TREs are the metadata payload of a NITF file: sensor models, block
descriptions, chipping parameters and several hundred registered others. They
appear in the user-defined and extended areas of any header, and spill into
data extension segments when they do not fit.

Each record is a 6-character tag, a 5-digit length and that many bytes. Only
the envelope is standardised, so this module always exposes the raw bytes and
decodes the contents only for tags it has a spec for. Register more with
`register`.
"""

import warnings

from .errors import NitfParseError
from .fields import Cursor

#: Tag -> callable(bytes) -> dict, for tags whose contents are decoded.
_PARSERS = {}


def register(tag, parser):
    """Register a decoder for a TRE tag.

    `parser` takes the record's raw bytes and returns a mapping. Registering a
    tag that is already known replaces the previous decoder.
    """
    _PARSERS[tag.strip().upper()] = parser


class TRE(object):
    """One tagged record.

    `data` is always the raw payload. `fields` decodes it when a parser is
    registered for the tag, and is None otherwise.
    """

    __slots__ = ("tag", "length", "data", "offset", "source", "_fields", "_parsed")

    def __init__(self, tag, length, data, offset=None, source=None):
        self.tag = tag
        self.length = length
        self.data = data
        #: Absolute file offset of the record's tag, when known.
        self.offset = offset
        #: Which header area the record came from, e.g. "IXSHD" or "UDHD".
        self.source = source
        self._fields = None
        self._parsed = False

    @property
    def fields(self):
        """Decoded contents, or None if the tag has no registered parser."""
        if not self._parsed:
            self._parsed = True
            parser = _PARSERS.get(self.tag.strip().upper())
            if parser is not None:
                self._fields = parser(self.data)
        return self._fields

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "<TRE %s %d bytes%s>" % (
            self.tag,
            len(self.data),
            "" if self.source is None else " from " + self.source,
        )


def parse_tres(data, source=None, base=0, strict=False):
    """Split a header extension area into tagged records.

    Lenient by default. A tagged record area is metadata sitting beside the
    pixel data, not part of the structure that locates it, so a malformed
    record costs the records after it and nothing else -- failing the whole
    read over one would be the wrong trade. Parsing stops at the first
    unusable record, keeps the ones already recovered, and warns. Pass
    ``strict=True`` to raise instead.
    """
    out = []
    pos = 0
    total = len(data)

    def give_up(message, offset):
        if strict:
            raise NitfParseError(message, offset=offset)
        warnings.warn(
            "%s; keeping the %d record(s) already read"
            % (message, len(out)),
            RuntimeWarning,
            stacklevel=3,
        )

    while pos < total:
        if total - pos < 11:
            give_up(
                "%d trailing bytes in %s are too few for a TRE header"
                % (total - pos, source or "extension area"),
                base + pos,
            )
            break
        cur = Cursor(data, start=pos, strict=False)
        tag = cur.text(6, "CETAG")
        length = cur.int(5, "CEL", default=None, allow_blank=True)
        if length is None:
            give_up(
                "TRE %r in %s has an unreadable length %r"
                % (tag, source or "extension area", cur.raw.get("CEL")),
                base + pos,
            )
            break
        start = cur.pos
        end = start + length
        if end > total:
            give_up(
                "TRE %s declares %d bytes but only %d remain in %s"
                % (tag, length, total - start, source or "extension area"),
                base + pos,
            )
            end = total
            out.append(TRE(tag, length, bytes(data[start:end]), base + pos, source))
            break
        out.append(TRE(tag, length, bytes(data[start:end]), base + pos, source))
        pos = end
    return out


class TRECollection(object):
    """The tagged records attached to one header, in file order.

    Indexing by tag is the common case, and several tags legitimately repeat,
    so `get` returns the first and `getall` returns every match.
    """

    def __init__(self, items=()):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._items[key]
        found = self.get(key)
        if found is None:
            raise KeyError(key)
        return found

    def __contains__(self, tag):
        return self.get(tag) is not None

    def __repr__(self):
        return "<TRECollection %s>" % ([t.tag for t in self._items],)

    @property
    def tags(self):
        return [t.tag for t in self._items]

    def get(self, tag, default=None):
        tag = tag.strip().upper()
        for item in self._items:
            if item.tag.strip().upper() == tag:
                return item
        return default

    def getall(self, tag):
        tag = tag.strip().upper()
        return [t for t in self._items if t.tag.strip().upper() == tag]

    def extend(self, items):
        self._items.extend(items)


def _spec_parser(spec):
    """Build a parser from a table of (name, length, kind) field triples."""

    def parse(data):
        cur = Cursor(data, strict=False)
        out = {}
        for name, length, kind in spec:
            if cur.remaining <= 0:
                break
            if kind == "n":
                out[name] = cur.int(length, name, default=None, allow_blank=True)
            elif kind == "f":
                out[name] = cur.float(length, name)
            else:
                out[name] = cur.text(length, name)
        return out

    return parse


#: BLOCKA: the true corner coordinates and row/column extent of an image
#: block, used where IGEOLO is too coarse or absent.
BLOCKA = (
    ("BLOCK_INSTANCE", 2, "n"),
    ("N_GRAY", 5, "n"),
    ("L_LINES", 5, "n"),
    ("LAYOVER_ANGLE", 3, "n"),
    ("SHADOW_ANGLE", 3, "n"),
    ("RESERVED1", 16, "a"),
    ("FRLC_LOC", 21, "a"),
    ("LRLC_LOC", 21, "a"),
    ("LRFC_LOC", 21, "a"),
    ("FRFC_LOC", 21, "a"),
    ("RESERVED2", 5, "a"),
)

#: ICHIPB: how a chip relates to the full image it was cut from, which is what
#: makes a sub-image's sensor model still usable.
ICHIPB = (
    ("XFRM_FLAG", 2, "n"),
    ("SCALE_FACTOR", 10, "f"),
    ("ANAMRPH_CORR", 2, "n"),
    ("SCANBLK_NUM", 2, "n"),
    ("OP_ROW_11", 12, "f"),
    ("OP_COL_11", 12, "f"),
    ("OP_ROW_12", 12, "f"),
    ("OP_COL_12", 12, "f"),
    ("OP_ROW_21", 12, "f"),
    ("OP_COL_21", 12, "f"),
    ("OP_ROW_22", 12, "f"),
    ("OP_COL_22", 12, "f"),
    ("FI_ROW_11", 12, "f"),
    ("FI_COL_11", 12, "f"),
    ("FI_ROW_12", 12, "f"),
    ("FI_COL_12", 12, "f"),
    ("FI_ROW_21", 12, "f"),
    ("FI_COL_21", 12, "f"),
    ("FI_ROW_22", 12, "f"),
    ("FI_COL_22", 12, "f"),
    ("FI_ROW", 8, "n"),
    ("FI_COL", 8, "n"),
)

register("BLOCKA", _spec_parser(BLOCKA))
register("ICHIPB", _spec_parser(ICHIPB))
