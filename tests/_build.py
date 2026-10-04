"""A minimal NITF writer, for tests only.

Building files in memory lets the reader be tested against pixels whose exact
layout is known, including combinations the sample corpus happens not to
contain. It writes only what the tests need and is not a general writer.
"""

import numpy as np

BLANK_FBKGC = b"\x00\x00\x00"


def a(text, length):
    """A text field, space padded on the right."""
    text = "" if text is None else str(text)
    if len(text) > length:
        raise ValueError("%r does not fit in %d characters" % (text, length))
    return text.ljust(length).encode("latin-1")


def n(value, length):
    """A numeric field, zero filled on the left."""
    text = "%0*d" % (length, int(value))
    if len(text) != length:
        raise ValueError("%r does not fit in %d digits" % (value, length))
    return text.encode("latin-1")


def security_21(prefix="U"):
    """A 167-byte unclassified 2.1 security block."""
    return (
        a(prefix, 1) + a("", 2) + a("", 11) + a("", 2) + a("", 20) + a("", 2)
        + a("", 8) + a("", 4) + a("", 1) + a("", 8) + a("", 43) + a("", 1)
        + a("", 40) + a("", 1) + a("", 8) + a("", 15)
    )


def security_20(prefix="U"):
    """A 167-byte unclassified 2.0 security block."""
    return (
        a(prefix, 1) + a("", 40) + a("", 40) + a("", 40) + a("", 20) + a("", 20)
        + a("", 6)
    )


def pack_bits(values, bits):
    """Pack integers MSB-first at `bits` each, padding the tail to a byte."""
    values = np.asarray(values).ravel()
    if bits == 12:
        out = bytearray()
        for i in range(0, values.size, 2):
            first = int(values[i]) & 0xFFF
            second = int(values[i + 1]) & 0xFFF if i + 1 < values.size else 0
            out += bytes(
                (first >> 4, ((first & 0xF) << 4) | (second >> 8), second & 0xFF)
            )
        return bytes(out[: (values.size * 12 + 7) // 8])
    acc = 0
    nbits = 0
    out = bytearray()
    for value in values:
        acc = (acc << bits) | (int(value) & ((1 << bits) - 1))
        nbits += bits
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def encode_pixels(array, imode, nbpr, nbpc, nppbh, nppbv, bits=None):
    """Serialise an (rows, cols, bands) array into NITF block order."""
    array = np.atleast_3d(array)
    rows, cols, bands = array.shape
    padded = np.zeros((nbpc * nppbv, nbpr * nppbh, bands), dtype=array.dtype)
    padded[:rows, :cols, :] = array

    def block(brow, bcol):
        return padded[
            brow * nppbv : (brow + 1) * nppbv, bcol * nppbh : (bcol + 1) * nppbh, :
        ]

    units = []
    if imode == "S":
        for band in range(bands):
            for brow in range(nbpc):
                for bcol in range(nbpr):
                    units.append(block(brow, bcol)[:, :, band].ravel())
    else:
        for brow in range(nbpc):
            for bcol in range(nbpr):
                b = block(brow, bcol)
                if imode == "B":
                    units.append(np.moveaxis(b, -1, 0).ravel())
                elif imode == "P":
                    units.append(b.ravel())
                elif imode == "R":
                    units.append(np.moveaxis(b, -1, 1).ravel())
                else:
                    raise ValueError("unsupported IMODE %r" % imode)

    # NITF pixels are big-endian whatever the host order is.
    big = array.dtype.newbyteorder(">")
    out = []
    for unit in units:
        if bits:
            out.append(pack_bits(unit, bits))
        else:
            out.append(unit.astype(big).tobytes())
    return units, b"".join(out)


def image_subheader_21(
    rows,
    cols,
    bands,
    pvtype="INT",
    irep="MONO",
    nbpp=8,
    abpp=None,
    imode="B",
    nbpr=1,
    nbpc=1,
    nppbh=None,
    nppbv=None,
    ic="NC",
    irepband=None,
    luts=None,
    pjust="R",
    iloc="0000000000",
    icom=(),
    ixshd=b"",
):
    nppbh = cols if nppbh is None else nppbh
    nppbv = rows if nppbv is None else nppbv
    abpp = nbpp if abpp is None else abpp
    parts = [
        a("IM", 2),
        a("TEST", 10),
        a("", 14),
        a("", 17),
        a("built by tests", 80),
        security_21(),
        a("0", 1),
        a("", 42),
        n(rows, 8),
        n(cols, 8),
        a(pvtype, 3),
        a(irep, 8),
        a("VIS", 8),
        n(abpp, 2),
        a(pjust, 1),
        a("", 1),  # ICORDS blank: no geolocation
        n(len(icom), 1),
    ]
    for comment in icom:
        parts.append(a(comment, 80))
    parts.append(a(ic, 2))
    if ic not in ("NC", "NM"):
        parts.append(a("0.0", 4))
    parts.append(n(bands, 1))
    names = irepband or ([""] * bands if bands != 3 else ["R", "G", "B"])
    for band in range(bands):
        parts += [a(names[band], 2), a("", 6), a("N", 1), a("", 3)]
        table = None if luts is None else luts[band]
        if table is None:
            parts.append(n(0, 1))
        else:
            table = np.asarray(table, dtype=np.uint8)
            parts += [n(table.shape[0], 1), n(table.shape[1], 5)]
            parts += [row.tobytes() for row in table]
    parts += [
        a("0", 1),
        a(imode, 1),
        n(nbpr, 4),
        n(nbpc, 4),
        n(nppbh, 4),
        n(nppbv, 4),
        n(nbpp, 2),
        n(1, 3),
        n(0, 3),
        a(iloc, 10),
        a("1.0", 4),
        n(0, 5),
    ]
    if ixshd:
        parts += [n(len(ixshd) + 3, 5), n(0, 3), ixshd]
    else:
        parts.append(n(0, 5))
    return b"".join(parts)


def file_header_21(image_lengths, ftitle="test", udhd=b"", des_lengths=()):
    """Assemble a 2.1 file header; FL and HL are filled in by `build`."""
    parts = [
        a("NITF", 4),
        a("02.10", 5),
        n(3, 2),
        a("BF01", 4),
        a("TESTS", 10),
        a("20240101000000", 14),
        a(ftitle, 80),
        security_21(),
        n(0, 5),
        n(0, 5),
        a("0", 1),
        BLANK_FBKGC,
        a("tests", 24),
        a("", 18),
    ]
    head = b"".join(parts)
    tail = [n(len(image_lengths), 3)]
    for sub, data in image_lengths:
        tail += [n(sub, 6), n(data, 10)]
    tail += [n(0, 3), n(0, 3), n(0, 3)]  # NUMS, NUMX, NUMT
    tail.append(n(len(des_lengths), 3))
    for sub, data in des_lengths:
        tail += [n(sub, 4), n(data, 9)]
    tail.append(n(0, 3))  # NUMRES
    if udhd:
        tail += [n(len(udhd) + 3, 5), n(0, 3), udhd]
    else:
        tail.append(n(0, 5))
    tail.append(n(0, 5))  # XHDL
    return head, b"".join(tail)


def build(segments, ftitle="test", udhd=b"", des=()):
    """Assemble a complete NITF 2.1 file.

    `segments` is a sequence of (subheader_bytes, data_bytes); `des` is a
    sequence of (des_subheader_bytes, des_data_bytes).
    """
    lengths = [(len(sub), len(data)) for sub, data in segments]
    des_lengths = [(len(sub), len(data)) for sub, data in des]
    head, tail = file_header_21(lengths, ftitle, udhd, des_lengths=des_lengths)
    body = b"".join(sub + data for sub, data in segments)
    body += b"".join(sub + data for sub, data in des)
    # FL and HL are the only fields that depend on the assembled size.
    header_length = len(head) + 12 + 6 + len(tail)
    total = header_length + len(body)
    return head + n(total, 12) + n(header_length, 6) + tail + body


def simple_image(array, **kwargs):
    """Build a one-segment file holding `array`, returning (bytes, expected)."""
    array3 = np.atleast_3d(array)
    rows, cols, bands = array3.shape
    imode = kwargs.pop("imode", "B")
    nbpr = kwargs.pop("nbpr", 1)
    nbpc = kwargs.pop("nbpc", 1)
    nppbh = kwargs.pop("nppbh", None) or -(-cols // nbpr)
    nppbv = kwargs.pop("nppbv", None) or -(-rows // nbpc)
    bits = kwargs.pop("bits", None)
    nbpp = kwargs.pop("nbpp", bits or array3.dtype.itemsize * 8)
    _, data = encode_pixels(array3, imode, nbpr, nbpc, nppbh, nppbv, bits)
    sub = image_subheader_21(
        rows,
        cols,
        bands,
        imode=imode,
        nbpr=nbpr,
        nbpc=nbpc,
        nppbh=nppbh,
        nppbv=nppbv,
        nbpp=nbpp,
        **kwargs
    )
    return build([(sub, data)])


def image_subheader_20(
    rows,
    cols,
    bands,
    pvtype="INT",
    irep="MONO",
    nbpp=8,
    imode="B",
    nbpr=1,
    nbpc=1,
    nppbh=None,
    nppbv=None,
    ic="NC",
    icords="N",
    igeolo=None,
):
    """A NITF 2.0 image subheader.

    The 2.0 layout differs from 2.1 in its security block and, awkwardly, in
    ICORDS: here "N" means no coordinates at all, where 2.1 uses it for UTM
    north and spells "none" as a blank.
    """
    nppbh = cols if nppbh is None else nppbh
    nppbv = rows if nppbv is None else nppbv
    parts = [
        a("IM", 2),
        a("TEST", 10),
        a("", 14),
        a("", 17),
        a("built by tests", 80),
        security_20(),
        a("0", 1),
        a("", 42),
        n(rows, 8),
        n(cols, 8),
        a(pvtype, 3),
        a(irep, 8),
        a("VIS", 8),
        n(nbpp, 2),
        a("R", 1),
        a(icords, 1),
    ]
    if igeolo is not None:
        parts.append(a(igeolo, 60))
    parts += [n(0, 1), a(ic, 2)]
    if ic not in ("NC", "NM"):
        parts.append(a("0.0", 4))
    parts.append(n(bands, 1))
    names = [""] * bands if bands != 3 else ["R", "G", "B"]
    for band in range(bands):
        parts += [a(names[band], 2), a("", 6), a("N", 1), a("", 3), n(0, 1)]
    parts += [
        a("0", 1),
        a(imode, 1),
        n(nbpr, 4),
        n(nbpc, 4),
        n(nppbh, 4),
        n(nppbv, 4),
        n(nbpp, 2),
        n(1, 3),
        n(0, 3),
        a("0000000000", 10),
        a("1.0", 4),
        n(0, 5),
        n(0, 5),
    ]
    return b"".join(parts)


def build_20(segments, ftitle="test"):
    """Assemble a complete NITF 2.0 file."""
    head = b"".join([
        a("NITF", 4),
        a("02.00", 5),
        n(3, 2),
        a("BF01", 4),
        a("TESTS", 10),
        a("01000000ZJAN24", 14),
        a(ftitle, 80),
        security_20(),
        n(0, 5),
        n(0, 5),
        a("0", 1),
        a("tests", 27),   # 2.0 ONAME is 27 characters and there is no FBKGC
        a("", 18),
    ])
    tail = [n(len(segments), 3)]
    for sub, data in segments:
        tail += [n(len(sub), 6), n(len(data), 10)]
    # NUMS, NUML (2.0 label segments), NUMT, NUMDES, NUMRES
    tail += [n(0, 3), n(0, 3), n(0, 3), n(0, 3), n(0, 3), n(0, 5), n(0, 5)]
    tail = b"".join(tail)
    body = b"".join(sub + data for sub, data in segments)
    header_length = len(head) + 12 + 6 + len(tail)
    total = header_length + len(body)
    return head + n(total, 12) + n(header_length, 6) + tail + body


def simple_image_20(array, **kwargs):
    """A one-segment NITF 2.0 file holding `array`."""
    array3 = np.atleast_3d(array)
    rows, cols, bands = array3.shape
    imode = kwargs.pop("imode", "B")
    _, data = encode_pixels(array3, imode, 1, 1, cols, rows)
    sub = image_subheader_20(
        rows, cols, bands, imode=imode, nbpp=array3.dtype.itemsize * 8, **kwargs
    )
    return build_20([(sub, data)])
