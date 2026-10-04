"""Read NITF imagery into numpy arrays.

NITF (National Imagery Transmission Format) packages imagery together with its
metadata. This package reads it in pure Python: the uncompressed path needs
nothing but numpy, and JPEG and JPEG 2000 segments are handed to Pillow when
it is installed.

    import vibenitf

    with vibenitf.open("scene.ntf") as f:
        print(f.header.ftitle)
        image = f.images[0]
        print(image.shape, image.dtype)
        pixels = image.read()
        corner = image.read(window=(0, 0, 512, 512))

`vibenitf.imread` collapses that to a single call when only the pixels matter.

Supported: NITF 2.1, NSIF 1.0 and NITF 2.0 headers; uncompressed imagery (IC
NC and NM) in every IMODE, with block masks, 1/2/4/12-bit packing, signed,
float and complex pixels, and lookup tables; JPEG (C3, M3, I1) and JPEG 2000
(C8, M8) through Pillow. Not supported: bi-level ITU-T.4 (C1, M1), vector
quantisation (C4, M4), lossless JPEG (C5, M5) and NITF 1.1 headers -- each
raises a specific error rather than returning wrong pixels.
"""

from . import codecs, fields, headers, image, masks, pixels, tre
from .errors import (
    NitfError,
    NitfParseError,
    NitfTruncatedError,
    UnsupportedCompressionError,
    UnsupportedFeatureError,
)
from .file import NitfFile, Segment, imread, open_nitf
from .headers import FileHeader, ImageSubheader
from .image import ImageSegment
from .masks import BlockMask
from .tre import TRE, TRECollection

__version__ = "0.1.0"

#: `vibenitf.open` is the primary entry point; `open_nitf` is the same
#: function under a name that cannot be mistaken for the builtin.
open = open_nitf

__all__ = [
    "open",
    "open_nitf",
    "imread",
    "NitfFile",
    "Segment",
    "ImageSegment",
    "FileHeader",
    "ImageSubheader",
    "BlockMask",
    "TRE",
    "TRECollection",
    "NitfError",
    "NitfParseError",
    "NitfTruncatedError",
    "UnsupportedCompressionError",
    "UnsupportedFeatureError",
    "codecs",
    "fields",
    "headers",
    "image",
    "masks",
    "pixels",
    "tre",
    "__version__",
]
