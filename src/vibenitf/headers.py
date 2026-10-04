"""NITF header and subheader structures.

Three header layouts exist in the wild. NITF 2.1 (MIL-STD-2500C) and NSIF 1.0
(STANAG 4545) are field-for-field identical and handled together; NITF 2.0
(MIL-STD-2500A) differs in the security block, the originator name length, the
absence of a background colour, and its use of label segments where 2.1 has a
reserved count. Those differences are isolated in the functions below rather
than spread through the readers.
"""

import re
import warnings

from .errors import NitfParseError, UnsupportedFeatureError
from .fields import Cursor

#: A version string that is at least shaped like one, e.g. 01.10.
_VERSION_SHAPE = re.compile(r"^\d\d\.\d\d$")

#: Bytes that count as padding at the end of a header.
BLANK = bytes((0x20, 0x09, 0x0D, 0x0A, 0x00))

V21 = "2.1"
V20 = "2.0"

#: Magic strings that open a file, mapped to the header layout they imply.
_MAGIC = {
    ("NITF", "02.10"): V21,
    ("NSIF", "01.00"): V21,
    ("NITF", "02.00"): V20,
}


def detect_version(first9):
    """Return the layout version implied by the first nine bytes of a file."""
    if len(first9) < 9:
        raise NitfParseError("file is too short to contain a magic number", offset=0)
    fhdr = first9[:4].decode("latin-1")
    fver = first9[4:9].decode("latin-1")
    try:
        return _MAGIC[(fhdr, fver)]
    except KeyError:
        pass
    if fhdr in ("NITF", "NSIF") and _VERSION_SHAPE.match(fver):
        raise UnsupportedFeatureError(
            "%s version %r is a real version but is not supported; this reader "
            "handles NITF 02.10, NSIF 01.00 and NITF 02.00" % (fhdr, fver)
        )
    raise NitfParseError(
        "not a NITF file: expected the magic NITF02.10, NSIF01.00 or NITF02.00 "
        "but found %r%r" % (fhdr, fver),
        offset=0,
        field="FHDR/FVER",
    )


class Header(object):
    """Base for parsed headers.

    Attributes are the lowercased NITF field names. `raw` maps every field
    name the parser consumed, in spec capitalisation, to its undecoded text,
    so nothing in the header is lost even where this package has no opinion
    about it.
    """

    def __repr__(self):
        name = type(self).__name__
        bits = []
        for attr in self._repr_fields:
            bits.append("%s=%r" % (attr, getattr(self, attr, None)))
        return "<%s %s>" % (name, " ".join(bits))

    _repr_fields = ()


def _security(cur, prefix, version):
    """Consume a security block, returning it as a dict.

    The 2.0 and 2.1 blocks are both 167 bytes, but share only their leading
    classification character; 2.0 adds a 40-byte declassification event when
    its downgrade field is the sentinel 999998.
    """
    out = {}

    def f(length, name, reader="text"):
        out[name.lower()] = getattr(cur, reader)(length, prefix + name)

    if version == V21:
        f(1, "CLAS")
        f(2, "CLSY")
        f(11, "CODE")
        f(2, "CTLH")
        f(20, "REL")
        f(2, "DCTP")
        f(8, "DCDT")
        f(4, "DCXM")
        f(1, "DG")
        f(8, "DGDT")
        f(43, "CLTX")
        f(1, "CATP")
        f(40, "CAUT")
        f(1, "CRSN")
        f(8, "SRDT")
        f(15, "CTLN")
    else:
        f(1, "CLAS")
        f(40, "CODE")
        f(40, "CTLH")
        f(40, "REL")
        f(20, "CAUT")
        f(20, "CTLN")
        f(6, "DWNG")
        if out["dwng"] == "999998":
            f(40, "DEVT")
    return out


def _extension_area(cur, len_name, ofl_name, length_size=5):
    """Consume a user-defined or extended header area.

    These areas are a length, and -- when the length is non-zero -- a three
    character overflow segment index followed by the tagged records
    themselves. The raw record bytes are returned for the TRE parser; a length
    of exactly 3 means the area holds the index but no records.
    """
    total = cur.int(length_size, len_name, default=0, allow_blank=True) or 0
    if total == 0:
        return 0, None, b""
    if total < 3:
        raise NitfParseError(
            "extension area length %d is too small to hold an overflow index" % total,
            offset=cur.offset,
            field=len_name,
        )
    # Writers disagree about whether the overflow index is present at all.
    # Peek before consuming: three digits or blanks is an index, anything else
    # is the first tagged record, and skipping it would lose three bytes and
    # misalign everything after.
    peek = bytes(cur.buf[cur.pos : cur.pos + 3])
    if peek.strip(BLANK).isdigit() or not peek.strip(BLANK):
        ofl = cur.int(3, ofl_name, default=0, allow_blank=True) or 0
        data = cur.take(total - 3, len_name + "_DATA")
    else:
        warnings.warn(
            "%s is %r, which is not an overflow segment index; treating the "
            "whole %d byte area as tagged records" % (ofl_name, peek, total),
            RuntimeWarning,
            stacklevel=3,
        )
        ofl = 0
        data = cur.take(total, len_name + "_DATA")
    return total, ofl, data


def _finish(cur, header, start, declared, field, strict):
    """Record how much of a header was consumed and police the declared length.

    Consuming more than was declared means the layout is wrong and everything
    after this header is misplaced, so that is fatal. Consuming less is not:
    the declared length still locates the next structure, nothing parsed is
    lost, and real writers do leave a few blank bytes at the end of a
    subheader. Those spare bytes are kept in `trailing_bytes`, and only
    non-blank ones are worth a warning.
    """
    header.parsed_length = cur.pos - start
    header.trailing_bytes = b""
    if not declared:
        return
    if header.parsed_length > declared:
        raise NitfParseError(
            "%s declares %d bytes but %d were consumed"
            % (field, declared, header.parsed_length),
            offset=start,
            field=field,
        )
    if header.parsed_length < declared:
        extra = bytes(cur.buf[start + header.parsed_length : start + declared])
        header.trailing_bytes = extra
        if extra.strip(BLANK) and strict:
            warnings.warn(
                "%d unparsed non-blank bytes at the end of a header declared by "
                "%s at offset %d" % (len(extra), field, start),
                RuntimeWarning,
                stacklevel=3,
            )


class SegmentLengths(object):
    """Declared subheader and data lengths for one segment."""

    __slots__ = ("subheader", "data")

    def __init__(self, subheader, data):
        self.subheader = subheader
        self.data = data

    def __repr__(self):
        return "SegmentLengths(subheader=%r, data=%r)" % (self.subheader, self.data)

    @property
    def total(self):
        return self.subheader + self.data


def _segment_list(cur, count_name, sub_size, data_size, count_max=None, strict=True):
    """Consume a segment count followed by that many length pairs."""
    count = cur.int(3, count_name, default=0, allow_blank=True) or 0
    if count_max is not None and count > count_max and strict:
        raise NitfParseError(
            "%s=%d exceeds the maximum of %d" % (count_name, count, count_max),
            field=count_name,
        )
    sub_name, data_name = _SEGMENT_FIELD_NAMES[count_name]
    out = []
    for i in range(1, count + 1):
        sub = cur.indexed(sub_size, sub_name, i, "int", default=0, allow_blank=True) or 0
        data = cur.indexed(data_size, data_name, i, "int", default=0, allow_blank=True) or 0
        out.append(SegmentLengths(sub, data))
    return out


#: Per-count-field names of the subheader-length and data-length subfields.
_SEGMENT_FIELD_NAMES = {
    "NUMI": ("LISH", "LI"),
    "NUMS": ("LSSH", "LS"),
    "NUML": ("LLSH", "LL"),
    "NUMT": ("LTSH", "LT"),
    "NUMDES": ("LDSH", "LD"),
    "NUMRES": ("LRESH", "LRE"),
}

#: An FL of all nines means the writer did not know the length in advance.
UNKNOWN_FILE_LENGTH = 999999999999


class FileHeader(Header):
    """The NITF file header: identification, security, and the segment table.

    The segment lists are the only index a NITF file has. Segment data is
    located by summing the declared lengths from the end of this header, which
    is why a single wrong length makes the remainder of a file unreadable.
    """

    _repr_fields = ("version", "ftitle", "numi", "file_length")

    @property
    def numi(self):
        return len(self.image_segments)

    @property
    def is_streaming(self):
        """True when FL was written as all nines (length not known up front).

        Such files normally carry an SSSHDR/streaming-file-header TRE giving
        the real values; the length must otherwise be taken from the file size.
        """
        return self.file_length == UNKNOWN_FILE_LENGTH


def parse_file_header(buf, strict=True):
    """Parse the file header at the start of `buf`."""
    version = detect_version(bytes(buf[:9]))
    cur = Cursor(buf, strict=strict)
    h = FileHeader()
    h.version = version
    h.offset = 0

    h.fhdr = cur.text(4, "FHDR")
    h.fver = cur.text(5, "FVER")
    h.clevel = cur.int(2, "CLEVEL", default=0, allow_blank=True)
    h.stype = cur.text(4, "STYPE")
    h.ostaid = cur.text(10, "OSTAID")
    h.fdt = cur.text(14, "FDT")
    h.ftitle = cur.text(80, "FTITLE")
    h.security = _security(cur, "FS", version)
    h.fscop = cur.int(5, "FSCOP", default=0, allow_blank=True)
    h.fscpys = cur.int(5, "FSCPYS", default=0, allow_blank=True)
    h.encryp = cur.int(1, "ENCRYP", default=0, allow_blank=True)
    if version == V21:
        h.fbkgc = tuple(cur.bytes(3, "FBKGC"))
        h.oname = cur.text(24, "ONAME")
    else:
        h.fbkgc = None
        h.oname = cur.text(27, "ONAME")
    h.ophone = cur.text(18, "OPHONE")
    h.file_length = cur.int(12, "FL")
    h.header_length = cur.int(6, "HL")

    h.image_segments = _segment_list(cur, "NUMI", 6, 10, 999, strict)
    h.graphic_segments = _segment_list(cur, "NUMS", 4, 6, 999, strict)
    if version == V21:
        # Reserved for future use and required to be zero; 2.0 put label
        # segments in this position instead.
        h.numx = cur.int(3, "NUMX", default=0, allow_blank=True)
        h.label_segments = []
    else:
        h.numx = None
        h.label_segments = _segment_list(cur, "NUML", 4, 3, 999, strict)
    h.text_segments = _segment_list(cur, "NUMT", 4, 5, 999, strict)
    h.des_segments = _segment_list(cur, "NUMDES", 4, 9, 999, strict)
    h.res_segments = _segment_list(cur, "NUMRES", 4, 7, 999, strict)

    h.udhdl, h.udhofl, h.udhd = _extension_area(cur, "UDHDL", "UDHOFL")
    h.xhdl, h.xhdofl, h.xhd = _extension_area(cur, "XHDL", "XHDOFL")

    h.raw = cur.raw
    _finish(cur, h, 0, h.header_length, "HL", strict)
    return h


class BandInfo(object):
    """One band record from an image subheader."""

    __slots__ = ("irepband", "isubcat", "ifc", "imflt", "nluts", "nelut", "luts")

    def __init__(self, irepband, isubcat, ifc, imflt, nluts, nelut, luts):
        self.irepband = irepband
        self.isubcat = isubcat
        self.ifc = ifc
        self.imflt = imflt
        self.nluts = nluts
        self.nelut = nelut
        self.luts = luts

    def __repr__(self):
        return "<BandInfo irepband=%r isubcat=%r nluts=%d>" % (
            self.irepband,
            self.isubcat,
            self.nluts,
        )


#: Compression codes that mean the pixels are stored uncompressed.
UNCOMPRESSED = ("NC", "NM")

#: Compression codes whose image data is preceded by a block/pad mask table.
MASKED = ("NM", "M1", "M3", "M4", "M5", "M8")


class ImageSubheader(Header):
    """The subheader describing one image segment.

    This records the declared geometry only. Interpreting it -- block layout,
    band interleave, pixel packing -- is done by `vibenitf.image`.
    """

    _repr_fields = ("iid1", "nrows", "ncols", "nbands", "pvtype", "nbpp", "ic", "imode")

    @property
    def is_masked(self):
        """True when a block mask table precedes the pixel data."""
        return self.ic in MASKED

    @property
    def is_compressed(self):
        return self.ic not in UNCOMPRESSED

    @property
    def pixels_per_block_h(self):
        """Block width in pixels, resolving the 0000 single-block encoding.

        NPPBH may be written as zero when the image is a single block wider
        than the 8192 the field can express; the block then spans the image.
        """
        if self.nppbh == 0:
            return self.ncols if self.nbpr == 1 else 0
        return self.nppbh

    @property
    def pixels_per_block_v(self):
        if self.nppbv == 0:
            return self.nrows if self.nbpc == 1 else 0
        return self.nppbv

    @property
    def block_count(self):
        return self.nbpr * self.nbpc


def parse_image_subheader(buf, start, version, strict=True, expected_length=None):
    """Parse the image subheader beginning at absolute offset `start`."""
    cur = Cursor(buf, start=start, strict=strict)
    h = ImageSubheader()
    h.version = version
    h.offset = start

    h.im = cur.text(2, "IM")
    if strict and h.im != "IM":
        raise NitfParseError(
            "image subheader does not start with 'IM' (found %r)" % h.im,
            offset=start,
            field="IM",
        )
    h.iid1 = cur.text(10, "IID1")
    h.idatim = cur.text(14, "IDATIM")
    h.tgtid = cur.text(17, "TGTID")
    h.iid2 = cur.text(80, "IID2")
    h.security = _security(cur, "IS", version)
    h.encryp = cur.int(1, "ENCRYP", default=0, allow_blank=True)
    h.isorce = cur.text(42, "ISORCE")
    h.nrows = cur.int(8, "NROWS")
    h.ncols = cur.int(8, "NCOLS")
    h.pvtype = cur.text(3, "PVTYPE")
    h.irep = cur.text(8, "IREP")
    h.icat = cur.text(8, "ICAT")
    h.abpp = cur.int(2, "ABPP")
    h.pjust = cur.text(1, "PJUST") or "R"

    h.icords = cur.text(1, "ICORDS")
    # The sentinel for "no coordinates" moved between versions: 2.0 spells it
    # N, while in 2.1 N means UTM north and a blank means none.
    if version == V20:
        has_geo = h.icords not in ("", "N")
    else:
        has_geo = h.icords != ""
    h.igeolo = cur.text(60, "IGEOLO") if has_geo else None

    h.nicom = cur.int(1, "NICOM", default=0, allow_blank=True) or 0
    h.icom = [cur.indexed(80, "ICOM", i) for i in range(1, h.nicom + 1)]

    h.ic = cur.text(2, "IC")
    h.comrat = cur.text(4, "COMRAT") if h.ic not in UNCOMPRESSED else None

    nbands = cur.int(1, "NBANDS", default=0, allow_blank=True) or 0
    if nbands == 0 and version == V21:
        # 2.1 escapes band counts above 9 through XBANDS; 2.0 cannot.
        nbands = cur.int(5, "XBANDS", default=0, allow_blank=True) or 0
        h.xbands = nbands
    else:
        h.xbands = None
    h.nbands = nbands

    h.bands = []
    for i in range(1, nbands + 1):
        irepband = cur.indexed(2, "IREPBAND", i)
        isubcat = cur.indexed(6, "ISUBCAT", i)
        ifc = cur.indexed(1, "IFC", i)
        imflt = cur.indexed(3, "IMFLT", i)
        nluts = cur.indexed(1, "NLUTS", i, "int", default=0, allow_blank=True) or 0
        nelut, luts = 0, []
        if nluts:
            nelut = cur.indexed(5, "NELUT", i, "int", default=0, allow_blank=True) or 0
            for j in range(1, nluts + 1):
                luts.append(bytes(cur.take(nelut, "LUTD%d_%d" % (i, j))))
        h.bands.append(BandInfo(irepband, isubcat, ifc, imflt, nluts, nelut, luts))

    h.isync = cur.int(1, "ISYNC", default=0, allow_blank=True)
    h.imode = cur.text(1, "IMODE") or "B"
    h.nbpr = cur.int(4, "NBPR", default=1, allow_blank=True) or 1
    h.nbpc = cur.int(4, "NBPC", default=1, allow_blank=True) or 1
    h.nppbh = cur.int(4, "NPPBH", default=0, allow_blank=True) or 0
    h.nppbv = cur.int(4, "NPPBV", default=0, allow_blank=True) or 0
    h.nbpp = cur.int(2, "NBPP")
    h.idlvl = cur.int(3, "IDLVL", default=0, allow_blank=True)
    h.ialvl = cur.int(3, "IALVL", default=0, allow_blank=True)
    h.iloc = cur.text(10, "ILOC")
    h.imag = cur.text(4, "IMAG")

    h.udidl, h.udofl, h.udid = _extension_area(cur, "UDIDL", "UDOFL")
    h.ixshdl, h.ixsofl, h.ixshd = _extension_area(cur, "IXSHDL", "IXSOFL")

    h.raw = cur.raw
    _finish(cur, h, start, expected_length, "LISH", strict)
    return h


def image_location(iloc):
    """Split an ILOC/SLOC field into (row, column) offsets.

    The field is two signed five-character integers; blanks mean zero.
    """
    text = (iloc or "").ljust(10)
    def part(s):
        s = s.strip()
        if not s or s == "-":
            return 0
        try:
            return int(s, 10)
        except ValueError:
            return 0
    return part(text[:5]), part(text[5:10])


class TextSubheader(Header):
    """Subheader of a text segment."""

    _repr_fields = ("textid", "txtitl", "txtfmt")


def parse_text_subheader(buf, start, version, strict=True, expected_length=None):
    cur = Cursor(buf, start=start, strict=strict)
    h = TextSubheader()
    h.version = version
    h.offset = start
    h.te = cur.text(2, "TE")
    if version == V21:
        h.textid = cur.text(7, "TEXTID")
        h.txtalvl = cur.int(3, "TXTALVL", default=0, allow_blank=True)
    else:
        h.textid = cur.text(10, "TEXTID")
        h.txtalvl = None
    h.txtdt = cur.text(14, "TXTDT")
    h.txtitl = cur.text(80, "TXTITL")
    h.security = _security(cur, "TS", version)
    h.encryp = cur.int(1, "ENCRYP", default=0, allow_blank=True)
    h.txtfmt = cur.text(3, "TXTFMT")
    h.txshdl, h.txsofl, h.txshd = _extension_area(cur, "TXSHDL", "TXSOFL")
    h.raw = cur.raw
    _finish(cur, h, start, expected_length, "LTSH", strict)
    return h


class GraphicSubheader(Header):
    """Subheader of a graphic segment, or of a 2.0 symbol segment."""

    _repr_fields = ("sid", "sname", "sfmt")


def parse_graphic_subheader(buf, start, version, strict=True, expected_length=None):
    cur = Cursor(buf, start=start, strict=strict)
    h = GraphicSubheader()
    h.version = version
    h.offset = start
    h.sy = cur.text(2, "SY")
    h.sid = cur.text(10, "SID")
    h.sname = cur.text(20, "SNAME")
    h.security = _security(cur, "SS", version)
    h.encryp = cur.int(1, "ENCRYP", default=0, allow_blank=True)
    if version == V21:
        # 2.1 graphics are always CGM, described only by a bounding box.
        h.sfmt = cur.text(1, "SFMT")
        h.sstruct = cur.text(13, "SSTRUCT")
        h.sdlvl = cur.int(3, "SDLVL", default=0, allow_blank=True)
        h.salvl = cur.int(3, "SALVL", default=0, allow_blank=True)
        h.sloc = cur.text(10, "SLOC")
        h.sbnd1 = cur.text(10, "SBND1")
        h.scolor = cur.text(1, "SCOLOR")
        h.sbnd2 = cur.text(10, "SBND2")
        h.sres2 = cur.text(2, "SRES2")
    else:
        # 2.0 symbols carry their own raster geometry and optional palette.
        h.sfmt = h.stype = cur.text(1, "STYPE")
        h.nlips = cur.int(4, "NLIPS", default=0, allow_blank=True)
        h.npixpl = cur.int(4, "NPIXPL", default=0, allow_blank=True)
        h.nwdth = cur.int(4, "NWDTH", default=0, allow_blank=True)
        h.nbpp = cur.int(1, "SNBPP", default=0, allow_blank=True)
        h.sdlvl = cur.int(3, "SDLVL", default=0, allow_blank=True)
        h.salvl = cur.int(3, "SALVL", default=0, allow_blank=True)
        h.sloc = cur.text(10, "SLOC")
        h.sloc2 = cur.text(10, "SLOC2")
        h.scolor = cur.text(1, "SCOLOR")
        h.snum = cur.text(6, "SNUM")
        h.srot = cur.text(3, "SROT")
        nelut = cur.int(3, "NELUT", default=0, allow_blank=True) or 0
        h.nelut = nelut
        h.lut = bytes(cur.take(nelut * 3, "LUTD")) if nelut else b""
    h.sxshdl, h.sxsofl, h.sxshd = _extension_area(cur, "SXSHDL", "SXSOFL")
    h.raw = cur.raw
    _finish(cur, h, start, expected_length, "LSSH", strict)
    return h


class LabelSubheader(Header):
    """Subheader of a NITF 2.0 label segment, which 2.1 removed."""

    _repr_fields = ("lid",)


def parse_label_subheader(buf, start, version, strict=True, expected_length=None):
    cur = Cursor(buf, start=start, strict=strict)
    h = LabelSubheader()
    h.version = version
    h.offset = start
    h.la = cur.text(2, "LA")
    h.lid = cur.text(10, "LID")
    h.security = _security(cur, "LS", version)
    h.encryp = cur.int(1, "ENCRYP", default=0, allow_blank=True)
    h.lfs = cur.text(1, "LFS")
    h.lcw = cur.int(2, "LCW", default=0, allow_blank=True)
    h.lch = cur.int(2, "LCH", default=0, allow_blank=True)
    h.ldlvl = cur.int(3, "LDLVL", default=0, allow_blank=True)
    h.lalvl = cur.int(3, "LALVL", default=0, allow_blank=True)
    h.lloc = cur.text(10, "LLOC")
    h.ltc = tuple(cur.bytes(3, "LTC"))
    h.lbc = tuple(cur.bytes(3, "LBC"))
    h.lxshdl, h.lxsofl, h.lxshd = _extension_area(cur, "LXSHDL", "LXSOFL")
    h.raw = cur.raw
    _finish(cur, h, start, expected_length, "LLSH", strict)
    return h


#: DESID values that mark a data extension holding overflowed tagged records.
TRE_OVERFLOW_IDS = ("TRE_OVERFLOW", "REGISTERED EXTENSIONS", "CONTROLLED EXTENSIONS")


class DESSubheader(Header):
    """Subheader of a data extension segment."""

    _repr_fields = ("desid", "desver", "desoflw", "desitem")

    @property
    def is_tre_overflow(self):
        """True when this segment holds records too large for their header."""
        return self.desid.strip().upper() in TRE_OVERFLOW_IDS


def parse_des_subheader(buf, start, version, strict=True, expected_length=None):
    cur = Cursor(buf, start=start, strict=strict)
    h = DESSubheader()
    h.version = version
    h.offset = start
    h.de = cur.text(2, "DE")
    h.desid = cur.text(25, "DESID")
    h.desver = cur.int(2, "DESVER", default=0, allow_blank=True)
    h.security = _security(cur, "DES", version)
    if h.desid.strip().upper() in TRE_OVERFLOW_IDS:
        # The overflow segment names the header whose records spilled here.
        h.desoflw = cur.text(6, "DESOFLW")
        h.desitem = cur.int(3, "DESITEM", default=0, allow_blank=True)
    else:
        h.desoflw = None
        h.desitem = None
    h.desshl = cur.int(4, "DESSHL", default=0, allow_blank=True) or 0
    h.desshf = bytes(cur.take(h.desshl, "DESSHF")) if h.desshl else b""
    h.raw = cur.raw
    _finish(cur, h, start, expected_length, "LDSH", strict)
    return h


class RESSubheader(Header):
    """Subheader of a reserved extension segment."""

    _repr_fields = ("resid", "resver")


def parse_res_subheader(buf, start, version, strict=True, expected_length=None):
    cur = Cursor(buf, start=start, strict=strict)
    h = RESSubheader()
    h.version = version
    h.offset = start
    h.re = cur.text(2, "RE")
    h.resid = cur.text(25, "RESID")
    h.resver = cur.int(2, "RESVER", default=0, allow_blank=True)
    h.security = _security(cur, "RES", version)
    h.resshl = cur.int(4, "RESSHL", default=0, allow_blank=True) or 0
    h.resshf = bytes(cur.take(h.resshl, "RESSHF")) if h.resshl else b""
    h.raw = cur.raw
    _finish(cur, h, start, expected_length, "LRESH", strict)
    return h


#: Delimiters bracketing the replacement header inside a streaming file
#: header data extension.
SFH_DELIM1 = b"\x0a\x6e\x1d\x97"
SFH_DELIM2 = b"\x0e\xca\x14\xbf"


def parse_streaming_file_header(data, strict=True):
    """Extract the replacement file header from a STREAMING_FILE_HEADER segment.

    The segment's data is not a bare header: it is a 7-digit length, a binary
    delimiter, the header itself, a closing delimiter, and the length again.
    The repeated length lets a reader find the header by seeking backwards from
    the end of a file it is still receiving.
    """
    cur = Cursor(data, strict=strict)
    length = cur.int(7, "SFH_L1")
    opening = bytes(cur.take(4, "SFH_DELIM1"))
    if opening != SFH_DELIM1:
        raise NitfParseError(
            "expected the streaming header delimiter %s but found %s"
            % (SFH_DELIM1.hex(), opening.hex()),
            offset=7,
            field="SFH_DELIM1",
        )
    body = bytes(cur.take(length, "SFH_DR"))
    closing = bytes(cur.take(4, "SFH_DELIM2"))
    if closing != SFH_DELIM2 and strict:
        raise NitfParseError(
            "expected the streaming header delimiter %s but found %s"
            % (SFH_DELIM2.hex(), closing.hex()),
            offset=cur.pos - 4,
            field="SFH_DELIM2",
        )
    repeated = cur.int(7, "SFH_L2", default=length, allow_blank=True)
    if repeated != length and strict:
        raise NitfParseError(
            "streaming header lengths disagree: SFH_L1=%d SFH_L2=%d"
            % (length, repeated),
            field="SFH_L2",
        )
    return parse_file_header(body, strict=strict)
