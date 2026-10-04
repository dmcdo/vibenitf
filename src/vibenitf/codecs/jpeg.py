"""JPEG-compressed image segments (IC values C3, M3 and I1).

Each block is an independent JPEG stream. Masked segments give the offset of
every stream in the block mask; unmasked blocked segments simply concatenate
them, so the Nth block has to be found by counting SOI markers.

Decoding is delegated to Pillow. Note that NITF permits 12-bit JPEG, which
libjpeg as built for Pillow cannot read -- that case is reported rather than
silently mangled.
"""

import io

import numpy as np

from ..errors import UnsupportedCompressionError
from . import _support

#: JPEG start-of-image marker.
SOI = b"\xff\xd8"


def available():
    try:
        from PIL import features
    except ImportError:
        return False
    return bool(features.check("jpg"))


def _require_pillow(ic):
    if not available():
        raise UnsupportedCompressionError(
            ic,
            "JPEG decoding needs Pillow with JPEG support; install it with "
            "'pip install vibenitf[jpeg]'",
        )


def _decode_stream(segment, start, end):
    from PIL import Image, UnidentifiedImageError

    data = _support.trim_to_marker(
        bytes(segment.buf[start:end]), SOI, "JPEG", segment.subheader.ic
    )
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return np.asarray(im)
    except (UnidentifiedImageError, OSError) as exc:
        if segment.subheader.nbpp == 12 or segment.subheader.abpp == 12:
            raise UnsupportedCompressionError(
                segment.subheader.ic,
                "12-bit JPEG at offset %d could not be decoded; Pillow's "
                "libjpeg is built for 8-bit samples (%s)" % (start, exc),
            ) from exc
        raise UnsupportedCompressionError(
            segment.subheader.ic,
            "JPEG stream at offset %d could not be decoded (%s)" % (start, exc),
        ) from exc


def decode_block(segment, brow, bcol, band_indices):
    ic = segment.subheader.ic
    _require_pillow(ic)
    bh, bw = segment.block_shape
    nbands = segment.bands
    ranges = _support.unit_ranges(segment, SOI)

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
