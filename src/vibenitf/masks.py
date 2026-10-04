"""The block and pad-pixel mask table that precedes masked image data.

When IC is one of the masked codes (NM, M1, M3, M4, M5, M8) the image segment
opens with a short table instead of pixels. The table gives the offset of every
block, with 0xFFFFFFFF marking a block that was never written -- the mechanism
NITF uses for sparse imagery and for transparent regions.
"""

import struct

import numpy as np

from .errors import NitfParseError

#: Offset value marking a block that is absent from the file.
ABSENT = 0xFFFFFFFF


class BlockMask(object):
    """Parsed mask table for one image segment."""

    def __init__(
        self,
        imdatoff,
        bmrlnth,
        tmrlnth,
        tpxcdlnth,
        tpxcd,
        block_offsets,
        pad_offsets,
        table_size,
    ):
        self.imdatoff = imdatoff
        self.bmrlnth = bmrlnth
        self.tmrlnth = tmrlnth
        self.tpxcdlnth = tpxcdlnth
        #: Raw pad-pixel code, or None when the image declares no pad value.
        self.tpxcd = tpxcd
        #: Per-block byte offsets relative to the start of the pixel data,
        #: or None when blocks are contiguous and can be addressed by index.
        self.block_offsets = block_offsets
        #: Per-block offsets into the pad-pixel mask, or None when absent.
        self.pad_offsets = pad_offsets
        #: Bytes consumed by the table as parsed.
        self.table_size = table_size

    def __repr__(self):
        return "<BlockMask imdatoff=%d bmrlnth=%d tmrlnth=%d blocks=%s>" % (
            self.imdatoff,
            self.bmrlnth,
            self.tmrlnth,
            "n/a" if self.block_offsets is None else len(self.block_offsets),
        )

    def offset_of(self, index):
        """Byte offset of block `index`, or None when the block is absent."""
        if self.block_offsets is None:
            return None
        value = int(self.block_offsets[index])
        return None if value == ABSENT else value

    def is_absent(self, index):
        return self.block_offsets is not None and (
            int(self.block_offsets[index]) == ABSENT
        )

    def has_pad(self, index):
        """True when block `index` is flagged as containing pad pixels."""
        if self.pad_offsets is None:
            return False
        return int(self.pad_offsets[index]) != ABSENT

    def pad_value(self, dtype, bands):
        """Decode TPXCD into a value comparable against decoded pixels.

        The code is stored as the raw bit pattern of one pixel -- which for a
        multi-band image is one value per band -- padded up to whole bytes.
        """
        if not self.tpxcd:
            return None
        raw = self.tpxcd
        try:
            if dtype.itemsize * bands == len(raw):
                values = np.frombuffer(raw, dtype=dtype)
            elif dtype.itemsize <= len(raw):
                values = np.frombuffer(raw[: dtype.itemsize], dtype=dtype)
            else:
                # Sub-byte pixel: the code sits in the low bits of one byte.
                values = np.array([raw[0]], dtype=dtype)
        except ValueError:
            return None
        return values if values.size > 1 else values[0]


def block_record_count(subheader):
    """Number of mask records, which is per band for band-sequential images."""
    count = subheader.nbpr * subheader.nbpc
    if subheader.imode == "S":
        count *= max(subheader.nbands, 1)
    return count


def parse_block_mask(buf, start, subheader, strict=True):
    """Parse the mask table at absolute offset `start`."""
    head = bytes(buf[start : start + 10])
    if len(head) < 10:
        raise NitfParseError(
            "image segment ends inside its mask table", offset=start, field="IMDATOFF"
        )
    imdatoff, bmrlnth, tmrlnth, tpxcdlnth = struct.unpack(">IHHH", head)
    pos = start + 10

    tpxcd = None
    if tpxcdlnth:
        nbytes = (tpxcdlnth + 7) // 8
        tpxcd = bytes(buf[pos : pos + nbytes])
        if len(tpxcd) != nbytes:
            raise NitfParseError(
                "image segment ends inside TPXCD", offset=pos, field="TPXCD"
            )
        pos += nbytes

    count = block_record_count(subheader)

    def records(length, name):
        nonlocal pos
        if not length:
            return None
        if length != 4 and strict:
            raise NitfParseError(
                "%s=%d; only 0 or 4 byte mask records are defined" % (name, length),
                offset=start,
                field=name,
            )
        nbytes = count * length
        chunk = bytes(buf[pos : pos + nbytes])
        if len(chunk) != nbytes:
            raise NitfParseError(
                "image segment ends inside %s" % name, offset=pos, field=name
            )
        pos += nbytes
        if length == 4:
            return np.frombuffer(chunk, dtype=">u4")
        return np.frombuffer(chunk, dtype=np.uint8).astype(">u4")

    block_offsets = records(bmrlnth, "BMRLNTH")
    pad_offsets = records(tmrlnth, "TMRLNTH")

    table_size = pos - start
    if strict and imdatoff and imdatoff < table_size:
        raise NitfParseError(
            "IMDATOFF=%d points inside the %d byte mask table"
            % (imdatoff, table_size),
            offset=start,
            field="IMDATOFF",
        )
    return BlockMask(
        imdatoff or table_size,
        bmrlnth,
        tmrlnth,
        tpxcdlnth,
        tpxcd,
        block_offsets,
        pad_offsets,
        table_size,
    )
