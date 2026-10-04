"""Helpers shared by the compressed-segment decoders."""

import warnings

import numpy as np


def segment_range(segment):
    """Absolute byte range of a segment's compressed data."""
    start = segment.pixel_offset
    end = segment.data_offset + segment.data_length
    if end <= start:
        end = len(segment.buf)
    return start, min(end, len(segment.buf))


def masked_ranges(segment):
    """Per-unit byte ranges taken from the block mask, or None if unmasked.

    Masked compressed blocks are variable length, so each unit runs from its
    own offset to the next offset present in the table. Absent units are None.
    """
    mask = segment.mask
    if mask is None or mask.block_offsets is None:
        return None
    start, end = segment_range(segment)
    offsets = [mask.offset_of(i) for i in range(len(mask.block_offsets))]
    present = sorted(o for o in offsets if o is not None)
    following = {}
    for i, offset in enumerate(present):
        following[offset] = present[i + 1] if i + 1 < len(present) else None
    out = []
    for offset in offsets:
        if offset is None:
            out.append(None)
            continue
        nxt = following[offset]
        out.append((start + offset, end if nxt is None else start + nxt))
    return out


def scan_streams(buf, start, end, marker, count):
    """Split a concatenated run of codestreams on their start marker.

    Blocked JPEG and JPEG 2000 segments store one stream per block back to
    back with no length table, so the only way to find block N is to count
    start markers. Stops once `count` streams are found.
    """
    data = bytes(buf[start:end])
    starts = []
    pos = 0
    while len(starts) < count:
        found = data.find(marker, pos)
        if found < 0:
            break
        starts.append(found)
        pos = found + len(marker)
    out = []
    for i, offset in enumerate(starts):
        tail = starts[i + 1] if i + 1 < len(starts) else len(data)
        out.append((start + offset, start + tail))
    return out


def unit_ranges(segment, marker):
    """Byte range per storage unit, from the mask table or by marker scan."""
    ranges = masked_ranges(segment)
    if ranges is not None:
        return ranges
    from ..masks import block_record_count

    count = block_record_count(segment.subheader)
    start, end = segment_range(segment)
    if count == 1:
        return [(start, end)]
    found = scan_streams(segment.buf, start, end, marker, count)
    if len(found) < count:
        warnings.warn(
            "expected %d codestreams in image segment %d but found %d"
            % (count, segment.index, len(found)),
            RuntimeWarning,
            stacklevel=2,
        )
        found += [None] * (count - len(found))
    return found


def trim_to_marker(data, marker, what, ic):
    """Drop any fill bytes preceding a codestream's start marker.

    NITF allows a block to be padded, and 0xFF is a legal fill byte, so a JPEG
    stream can sit behind a run of them. Decoders reject that, and the leading
    bytes carry no information, so they are simply skipped.
    """
    at = data.find(marker)
    if at < 0:
        from ..errors import UnsupportedCompressionError

        raise UnsupportedCompressionError(
            ic, "no %s start marker found in %s" % (what, "the block")
        )
    return data[at:] if at else data


def conform(array, bh, bw, nbands, band_indices, segment):
    """Shape a decoder's output into (bh, bw, len(band_indices)).

    Decoders may return a 2-D plane, or an image sized to the real pixel
    extent rather than the padded block, so pad and crop to the block before
    selecting bands.
    """
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[2] < nbands:
        # A greyscale stream standing in for a multi-band block: repeat it
        # rather than fail, which is what viewers do with such files.
        array = np.repeat(array[:, :, :1], nbands, axis=2)
    elif array.shape[2] > nbands:
        array = array[:, :, :nbands]

    if array.shape[0] != bh or array.shape[1] != bw:
        if array.shape[0] < bh or array.shape[1] < bw:
            padded = np.zeros((bh, bw, array.shape[2]), dtype=array.dtype)
            padded[: array.shape[0], : array.shape[1], :] = array[:bh, :bw, :]
            array = padded
        else:
            array = array[:bh, :bw, :]

    if band_indices != list(range(nbands)):
        array = array[:, :, band_indices]
    target = segment.dtype
    if array.dtype != target:
        array = array.astype(target, copy=False)
    return array
