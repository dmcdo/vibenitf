"""Pixel type resolution and bit unpacking.

NITF stores pixels big-endian, and allows field widths narrower than a byte
(1, 2 and 4 bit) as well as the awkward 12-bit case. Within a block, sub-byte
pixels are packed continuously most-significant-bit first; only the end of the
block is padded to a byte boundary. Nothing is padded per row.
"""

import numpy as np

from .errors import UnsupportedFeatureError

#: Field widths that do not land on a byte boundary and need unpacking.
PACKED_BITS = (1, 2, 4, 12)

#: Output container for each packed width.
_PACKED_DTYPE = {1: np.uint8, 2: np.uint8, 4: np.uint8, 12: np.uint16}

#: (PVTYPE, NBPP) -> big-endian numpy dtype for byte-aligned pixels.
_DIRECT = {
    ("INT", 8): ">u1",
    ("INT", 16): ">u2",
    ("INT", 32): ">u4",
    ("INT", 64): ">u8",
    ("SI", 8): ">i1",
    ("SI", 16): ">i2",
    ("SI", 32): ">i4",
    ("SI", 64): ">i8",
    ("R", 32): ">f4",
    ("R", 64): ">f8",
    ("C", 64): ">c8",
    ("C", 128): ">c16",
}


def resolve_dtype(pvtype, nbpp):
    """Map PVTYPE and NBPP to a numpy dtype and a packed bit width.

    Returns ``(dtype, packed_bits)``. `packed_bits` is None when pixels are
    byte aligned and can be read with a plain dtype; otherwise it is the field
    width in bits and `dtype` is the container the unpacked values land in.
    """
    pvtype = (pvtype or "INT").strip().upper()
    if pvtype == "B":
        # Bi-level. The standard fixes NBPP at 1, but be forgiving about
        # writers that pad the declaration out to a byte.
        if nbpp not in (1, 8):
            raise UnsupportedFeatureError(
                "PVTYPE=B with NBPP=%r; bi-level pixels must be 1 bit" % nbpp
            )
        return np.dtype(np.uint8), (1 if nbpp == 1 else None)

    if pvtype in ("INT", "SI") and nbpp in PACKED_BITS:
        if pvtype == "SI" and nbpp != 12:
            raise UnsupportedFeatureError(
                "PVTYPE=SI with NBPP=%d is not a defined combination" % nbpp
            )
        return np.dtype(_PACKED_DTYPE[nbpp]), nbpp

    try:
        return np.dtype(_DIRECT[(pvtype, nbpp)]), None
    except KeyError:
        raise UnsupportedFeatureError(
            "no pixel layout defined for PVTYPE=%r NBPP=%r" % (pvtype, nbpp)
        ) from None


def packed_size(count, bits):
    """Bytes occupied by `count` pixels of `bits` each, padded to a byte."""
    return (count * bits + 7) // 8


def unpack(buf, count, bits):
    """Unpack `count` MSB-first packed pixels of width `bits` from `buf`."""
    raw = np.frombuffer(buf, dtype=np.uint8)
    if bits == 12:
        return _unpack12(raw, count)
    if bits not in (1, 2, 4):
        raise UnsupportedFeatureError("cannot unpack %r bit pixels" % bits)
    bitarr = np.unpackbits(raw)
    per = bits
    usable = (bitarr.size // per) * per
    vals = bitarr[:usable].reshape(-1, per)
    if per == 1:
        out = vals.reshape(-1)
    else:
        # Weight the bits of each field, most significant first.
        weights = (1 << np.arange(per - 1, -1, -1)).astype(np.uint8)
        out = (vals * weights).sum(axis=1, dtype=np.uint8)
    if out.size < count:
        raise ValueError("buffer holds %d of %d pixels" % (out.size, count))
    return out[:count]


def _unpack12(raw, count):
    """Unpack 12-bit pixels, which occupy three bytes per two pixels."""
    triples = (count + 1) // 2
    needed = triples * 3
    if raw.size < packed_size(count, 12):
        raise ValueError(
            "buffer of %d bytes holds fewer than %d 12-bit pixels" % (raw.size, count)
        )
    if raw.size < needed:
        # An odd final pixel leaves the third byte of its triple unwritten.
        raw = np.concatenate([raw, np.zeros(needed - raw.size, np.uint8)])
    b = raw[:needed].reshape(triples, 3).astype(np.uint16)
    out = np.empty(triples * 2, dtype=np.uint16)
    out[0::2] = (b[:, 0] << 4) | (b[:, 1] >> 4)
    out[1::2] = ((b[:, 1] & 0x0F) << 8) | b[:, 2]
    return out[:count]


def apply_justification(array, abpp, nbpp, pjust):
    """Shift left-justified values down so they carry their true magnitude.

    When ABPP is narrower than NBPP the significant bits sit at one end of the
    field. PJUST=R (the usual case) needs nothing; PJUST=L means the value was
    shifted up into the high bits and must come back down.
    """
    if not abpp or not nbpp or abpp >= nbpp:
        return array
    if (pjust or "R").strip().upper() != "L":
        return array
    if array.dtype.kind not in "ui":
        return array
    return array >> np.array(nbpp - abpp, dtype=array.dtype)
