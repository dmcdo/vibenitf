"""Decoders for compressed image segments.

Everything in this package is optional. The uncompressed path in
`vibenitf.image` is pure Python and numpy; JPEG and JPEG 2000 are delegated to
Pillow, because writing either codec in Python would be both slow and a poor
use of anyone's trust. When a delegate is missing, or a scheme has no decoder
at all, the error says which IC value was involved and what would fix it.
"""

from ..errors import UnsupportedCompressionError

#: IC values with no decoder, and what each one would take to support.
UNIMPLEMENTED = {
    "C1": "bi-level ITU-T.4 (Group 3/4) compression is not implemented",
    "M1": "masked bi-level ITU-T.4 (Group 3/4) compression is not implemented",
    "C4": "vector quantisation (NITF/RPF VQ) is not implemented",
    "M4": "masked vector quantisation (NITF/RPF VQ) is not implemented",
    "C5": "lossless JPEG is not implemented; Pillow cannot decode it either",
    "M5": "masked lossless JPEG is not implemented",
    "C6": "C6 compression is reserved and has no public definition",
    "M6": "M6 compression is reserved and has no public definition",
    "C7": "C7 compression is reserved and has no public definition",
    "M7": "M7 compression is reserved and has no public definition",
}


def _decoder(ic):
    if ic in ("C3", "M3", "I1"):
        from . import jpeg

        return jpeg
    if ic in ("C8", "M8"):
        from . import j2k

        return j2k
    raise UnsupportedCompressionError(
        ic, UNIMPLEMENTED.get(ic, "unrecognised compression code")
    )


def decode_block(segment, brow, bcol, band_indices):
    """Decode one block of a compressed segment.

    Returns an array shaped (block_rows, block_cols, len(band_indices)), or
    None when the block is absent from the file.
    """
    return _decoder(segment.subheader.ic).decode_block(
        segment, brow, bcol, band_indices
    )


def is_supported(ic):
    """True when this build can decode image data compressed with `ic`."""
    if ic in ("NC", "NM"):
        return True
    try:
        return _decoder(ic).available()
    except UnsupportedCompressionError:
        return False
