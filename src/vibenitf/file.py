"""Opening a NITF file and walking its segments.

A NITF file has no central index. The file header declares, for each segment,
the length of its subheader and the length of its data; segments then follow
nose to tail in a fixed order. Positions are therefore cumulative, and the
walk below is the only way to find anything.
"""

import io
import mmap
import os

from . import headers, tre
from .errors import NitfParseError
from .image import ImageSegment


class Segment(object):
    """A non-image segment: its subheader, and where its data lives."""

    def __init__(self, kind, index, subheader, data_offset, data_length, buf):
        self.kind = kind
        self.index = index
        self.subheader = subheader
        self.data_offset = data_offset
        self.data_length = data_length
        self.buf = buf
        self.tres = tre.TRECollection()

    def read_bytes(self):
        """The segment's data, verbatim."""
        return bytes(self.buf[self.data_offset : self.data_offset + self.data_length])

    def read_text(self, encoding="latin-1"):
        """The segment's data decoded as text."""
        return self.read_bytes().decode(encoding)

    def __repr__(self):
        return "<%s segment %d %d bytes>" % (self.kind, self.index, self.data_length)


#: DESID of the data extension that carries a replacement file header.
STREAMING_FILE_HEADER = "STREAMING_FILE_HEADER"

#: Which header extension area each DESOFLW value redirects records into.
_OVERFLOW_TARGETS = {
    "UDHD": ("file", "udhd"),
    "XHD": ("file", "xhd"),
    "UDID": ("image", "udid"),
    "IXSHD": ("image", "ixshd"),
    "SXSHD": ("graphic", "sxshd"),
    "LXSHD": ("label", "lxshd"),
    "TXSHD": ("text", "txshd"),
}


class NitfFile(object):
    """An open NITF file.

    Use as a context manager, or call `close` when done; the file is memory
    mapped, so the mapping must be released before the file can be removed on
    Windows.
    """

    def __init__(self, buf, strict=True, closer=None):
        self.buf = buf
        self.strict = strict
        self._closer = closer
        self.header = headers.parse_file_header(buf, strict=strict)
        #: The as-written header when it was replaced by a streaming one.
        self.streamed_header = None
        self._load_streaming_header()

        self.images = []
        self.graphics = []
        self.labels = []
        self.texts = []
        self.data_extensions = []
        self.reserved_extensions = []
        self.tres = tre.TRECollection()

        self._walk()
        self._collect_tres()
        self._resolve_overflow()

    # -- construction ------------------------------------------------------

    def _load_streaming_header(self):
        """Replace an all-nines header with the real one from the file's tail.

        A writer streaming imagery cannot know FL or LI in advance, so it
        writes nines and appends a STREAMING_FILE_HEADER data extension
        holding a complete, correct copy of the file header. Without this
        substitution the segment lengths are unusable.
        """
        h = self.header
        if not h.is_streaming or not h.des_segments:
            return
        last = h.des_segments[-1]
        start = len(self.buf) - last.total
        if start <= 0:
            return
        try:
            sub = headers.parse_des_subheader(self.buf, start, h.version, strict=False)
        except NitfParseError:
            return
        if sub.desid.strip().upper() != STREAMING_FILE_HEADER:
            return
        data = bytes(
            self.buf[start + last.subheader : start + last.subheader + last.data]
        )
        try:
            replacement = headers.parse_streaming_file_header(data, strict=False)
        except NitfParseError:
            return
        if replacement.header_length != h.header_length:
            # The copy must describe the same layout to be substitutable.
            return
        self.streamed_header = h
        self.header = replacement

    def _walk(self):
        h = self.header
        offset = h.header_length or h.parsed_length
        version = h.version

        for i, lengths in enumerate(h.image_segments):
            sub = headers.parse_image_subheader(
                self.buf,
                offset,
                version,
                strict=self.strict,
                expected_length=lengths.subheader,
            )
            self.images.append(
                ImageSegment(
                    self.buf,
                    sub,
                    offset + lengths.subheader,
                    lengths.data,
                    index=i,
                    strict=self.strict,
                )
            )
            offset += lengths.total

        simple = (
            ("graphic", h.graphic_segments, headers.parse_graphic_subheader, self.graphics),
            ("label", h.label_segments, headers.parse_label_subheader, self.labels),
            ("text", h.text_segments, headers.parse_text_subheader, self.texts),
            ("des", h.des_segments, headers.parse_des_subheader, self.data_extensions),
            ("res", h.res_segments, headers.parse_res_subheader, self.reserved_extensions),
        )
        for kind, table, parse, target in simple:
            for i, lengths in enumerate(table):
                sub = parse(
                    self.buf,
                    offset,
                    version,
                    strict=self.strict,
                    expected_length=lengths.subheader,
                )
                target.append(
                    Segment(
                        kind,
                        i,
                        sub,
                        offset + lengths.subheader,
                        lengths.data,
                        self.buf,
                    )
                )
                offset += lengths.total

        self.end_offset = offset
        if self.strict and not h.is_streaming and h.file_length:
            if offset != h.file_length:
                raise NitfParseError(
                    "segment lengths sum to %d but FL declares %d"
                    % (offset, h.file_length),
                    field="FL",
                )

    def _collect_tres(self):
        """Attach the tagged records found in each header's extension areas."""
        # Tagged records are parsed leniently even in strict mode: they do not
        # take part in locating anything, so a bad record must not cost the
        # caller their pixels. The raw bytes stay on each header regardless.
        h = self.header
        self.tres.extend(tre.parse_tres(h.udhd, "UDHD"))
        self.tres.extend(tre.parse_tres(h.xhd, "XHD"))
        for segment in self.images:
            sub = segment.subheader
            segment.tres = tre.TRECollection(
                tre.parse_tres(sub.udid, "UDID")
                + tre.parse_tres(sub.ixshd, "IXSHD")
            )
        for segment in self.graphics:
            segment.tres = tre.TRECollection(
                tre.parse_tres(segment.subheader.sxshd, "SXSHD")
            )
        for segment in self.labels:
            segment.tres = tre.TRECollection(
                tre.parse_tres(segment.subheader.lxshd, "LXSHD")
            )
        for segment in self.texts:
            segment.tres = tre.TRECollection(
                tre.parse_tres(segment.subheader.txshd, "TXSHD")
            )

    def _resolve_overflow(self):
        """Fold records from TRE overflow segments back into their headers.

        A header's extension area is limited to 99999 bytes, so large records
        are moved into a data extension segment that names the header they
        belong to. Readers that skip this step silently lose metadata.
        """
        for des in self.data_extensions:
            sub = des.subheader
            if not sub.is_tre_overflow:
                continue
            target = _OVERFLOW_TARGETS.get((sub.desoflw or "").strip().upper())
            if target is None:
                continue
            scope, area = target
            records = tre.parse_tres(des.read_bytes(), area + " (overflow)")
            if not records:
                continue
            if scope == "file":
                self.tres.extend(records)
                continue
            collection = {
                "image": self.images,
                "graphic": self.graphics,
                "label": self.labels,
                "text": self.texts,
            }[scope]
            # DESITEM is a 1-based index into the segments of that type.
            position = (sub.desitem or 0) - 1
            if 0 <= position < len(collection):
                collection[position].tres.extend(records)

    # -- lifecycle ---------------------------------------------------------

    def close(self):
        if self._closer is not None:
            self._closer()
            self._closer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "<NitfFile %s %d image(s) %d text(s) %d des>" % (
            self.header.version,
            len(self.images),
            len(self.texts),
            len(self.data_extensions),
        )

    # -- convenience -------------------------------------------------------

    @property
    def version(self):
        return self.header.version

    def read(self, index=0, **kwargs):
        """Read image segment `index`; keyword arguments go to `ImageSegment.read`."""
        return self.images[index].read(**kwargs)


def open_nitf(source, strict=True):
    """Open a NITF file from a path, or wrap an in-memory buffer.

    strict
        When true, declared lengths are cross-checked and any inconsistency
        raises. Set it to false to read damaged files on a best-effort basis,
        which downgrades those checks to warnings where recovery is possible.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return NitfFile(source, strict=strict)

    path = os.fspath(source)
    handle = io.open(path, "rb")
    try:
        size = os.fstat(handle.fileno()).st_size
        if size == 0:
            raise NitfParseError("%s is empty" % path, offset=0)
        buf = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        handle.close()
        raise

    def closer():
        buf.close()
        handle.close()

    try:
        return NitfFile(buf, strict=strict, closer=closer)
    except Exception:
        closer()
        raise


def imread(source, index=0, **kwargs):
    """Read one image segment from a NITF file in a single call."""
    with open_nitf(source, strict=kwargs.pop("strict", True)) as nitf:
        return nitf.images[index].read(**kwargs)
