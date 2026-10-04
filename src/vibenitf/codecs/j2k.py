"""JPEG 2000 image segments (IC values C8 and M8).

JPEG 2000 does its own tiling, so a C8 segment normally holds one codestream
covering the whole image and the NITF block grid is decorative. M8 segments may
instead hold one codestream per block, addressed through the block mask. Both
arrangements occur, so this module looks for per-block streams first and falls
back to decoding the segment as a single image, slicing blocks out of it.

Decoding is delegated to Pillow's OpenJPEG binding.
"""

import io

import numpy as np

from ..errors import UnsupportedCompressionError
from ..masks import block_record_count
from . import _support

#: SOC followed by SIZ: the start of a raw JPEG 2000 codestream.
SOC = b"\xff\x4f\xff\x51"

#: First four bytes of the JP2 container signature box.
JP2_SIGNATURE = b"\x00\x00\x00\x0c"


def available():
    try:
        from PIL import features
    except ImportError:
        return False
    return bool(features.check("jpg_2000"))


def _require_pillow(ic):
    if not available():
        raise UnsupportedCompressionError(
            ic,
            "JPEG 2000 decoding needs Pillow with OpenJPEG support; install it "
            "with 'pip install vibenitf[jp2]'",
        )


def _decode_stream(segment, start, end):
    from PIL import Image, UnidentifiedImageError

    data = bytes(segment.buf[start:end])
    # A segment may hold either a raw codestream or a JP2 container; only the
    # former can have fill bytes in front of it worth skipping.
    if data[:4] != JP2_SIGNATURE:
        data = _support.trim_to_marker(data, SOC, "JPEG 2000", segment.subheader.ic)
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return np.asarray(im)
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedCompressionError(
            segment.subheader.ic,
            "JPEG 2000 codestream at offset %d could not be decoded (%s)"
            % (start, exc),
        ) from exc


def _whole_image(segment):
    """Decode and cache the segment as a single JPEG 2000 image."""
    cached = getattr(segment, "_j2k_whole", None)
    if cached is None:
        cached = _decode_stream(segment, *_support.segment_range(segment))
        if cached.ndim == 2:
            cached = cached[:, :, None]
        segment._j2k_whole = cached
    return cached


def _per_unit_ranges(segment):
    """Per-block codestream ranges, or None when the segment holds just one."""
    count = block_record_count(segment.subheader)
    if count == 1:
        return None
    ranges = _support.masked_ranges(segment)
    if ranges is not None:
        return ranges
    start, end = _support.segment_range(segment)
    found = _support.scan_streams(segment.buf, start, end, SOC, count)
    return found if len(found) >= count else None


def decode_block(segment, brow, bcol, band_indices):
    ic = segment.subheader.ic
    _require_pillow(ic)
    bh, bw = segment.block_shape
    nbands = segment.bands
    ranges = _per_unit_ranges(segment)

    if ranges is None:
        # One codestream for the image: cut the block out of the decoded whole.
        whole = _whole_image(segment)
        top, left = brow * bh, bcol * bw
        region = whole[top : top + bh, left : left + bw, :]
        if region.size == 0:
            return None
        return _support.conform(region, bh, bw, nbands, band_indices, segment)

    if segment.subheader.imode == "S":
        planes = []
        for band in band_indices:
            index = segment._unit_index(brow, bcol, band)
            span = ranges[index] if index < len(ranges) else None
            if span is None:
                return None
            plane = _decode_stream(segment, *span)
            planes.append(_support.conform(plane, bh, bw, 1, [0], segment))
        return np.concatenate(planes, axis=-1)

    index = segment._unit_index(brow, bcol)
    span = ranges[index] if index < len(ranges) else None
    if span is None:
        return None
    array = _decode_stream(segment, *span)
    return _support.conform(array, bh, bw, nbands, band_indices, segment)
