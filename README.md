# vibenitf

Read NITF imagery into numpy arrays, in pure Python.

NITF (National Imagery Transmission Format) packages imagery together with its
metadata. This package reads it with nothing but numpy for uncompressed
imagery, which is the large majority of NITF in practice. JPEG and JPEG 2000
segments are handed to Pillow when it is installed.

```python
import vibenitf

with vibenitf.open("scene.ntf") as f:
    print(f.header.ftitle)

    image = f.images[0]
    print(image.shape, image.dtype)     # (1024, 1024) uint8

    pixels = image.read()               # the whole segment
    corner = image.read(window=(0, 0, 512, 512))   # reads only the blocks it needs
    green  = image.read(bands=1)
    valid  = image.read_mask()          # False where a block was never written
```

One-liner when only the pixels matter:

```python
array = vibenitf.imread("scene.ntf")
```

## Install

```sh
pip install vibenitf          # uncompressed imagery: numpy only
pip install vibenitf[all]     # adds Pillow for JPEG and JPEG 2000 segments
```

## What it supports

| | |
|---|---|
| Headers | NITF 2.1 (MIL-STD-2500C), NSIF 1.0 (STANAG 4545), NITF 2.0 (MIL-STD-2500A) |
| Compression | `NC`, `NM` uncompressed (pure Python); `C3`, `M3`, `I1` JPEG and `C8`, `M8` JPEG 2000 via Pillow |
| Band interleave | `IMODE` B, P, R and S |
| Pixel types | `INT`, `SI`, `R`, `C`, `B` — 1, 2, 4, 8, 12, 16, 32 and 64 bits, big-endian, with `ABPP`/`PJUST` handling |
| Blocking | Arbitrary block grids, edge padding, the `NPPBH=0000` single-block encoding |
| Masks | Block masks, absent blocks, pad-pixel masks and `TPXCD` |
| Metadata | Palettes (`LUTD`), every subheader type, TREs including those overflowed into a data extension, streaming file headers |

Not supported, each raising a specific error rather than returning wrong
pixels:

- `C1`/`M1` bi-level ITU-T.4 (Group 3/4) compression
- `C4`/`M4` vector quantisation, `C5`/`M5` lossless JPEG, and the reserved
  `C6`, `C7` codes
- 12-bit JPEG, which Pillow's libjpeg cannot decode
- NITF 1.1 headers

## API

`vibenitf.open(path_or_bytes, strict=True)` returns a `NitfFile`:

- `.header` — the `FileHeader`. Every parsed field is an attribute under its
  lowercased NITF name, and `.raw` maps each spec field name to its undecoded
  text, so nothing in the header is lost.
- `.images` — `ImageSegment` objects. Also `.texts`, `.graphics`, `.labels`,
  `.data_extensions`, `.reserved_extensions`.
- `.tres` — file-level tagged records. Each segment has its own `.tres`.
- `.close()`, or use it as a context manager. The file is memory mapped, so
  the mapping must be released before the file can be deleted on Windows.

`ImageSegment.read()` takes:

| argument | meaning |
|---|---|
| `window` | `(row_start, col_start, row_stop, col_stop)`, half open, clipped to the image. Only the blocks it touches are read. |
| `bands` | A band index, a sequence of indices, or None for all. Order is honoured. |
| `band_axis` | `"last"` for `(rows, cols, bands)`, `"first"` for `(bands, rows, cols)`. |
| `squeeze` | Drop the band axis when one band is returned. Default True. |
| `justify` | Shift left-justified values down to their true magnitude when `ABPP < NBPP`. Default True. |
| `apply_lut` | Expand palettised values through the segment's lookup tables. Default False, so the raw indices are what you get. |
| `fill` | Value for pixels whose block is absent from the file. Default 0. |

Also `read_block(brow, bcol)` for one block including its padding, and
`read_mask()` for a bool array that is False where a block was never written
or a pixel equals the pad code.

`strict=False` relaxes cross-checks on declared lengths and pads truncated
blocks with zeros instead of raising, for reading damaged files.

## Command line

```sh
python -m vibenitf info scene.ntf --tres
python -m vibenitf export scene.ntf out.npy --window 0,0,512,512
python -m vibenitf export scene.ntf preview.png --lut
```

`info` prints the header, every segment's geometry and block layout, the mask
state, and warns when a segment's compression cannot be decoded. `export`
writes exact values to `.npy`, or a contrast-scaled preview to any format
Pillow can write.

## Tests

```sh
pip install -e .[test]
python tools/fetch_samples.py     # ~62 MB of public sample files
python -m pytest
```

`tests/test_synthetic.py` builds NITF files in memory with a small writer, so
pixel layouts are known exactly and every `IMODE`, packing width and mask
arrangement can be checked without any download.

`tests/test_corpus.py` runs against the JITC NITF 2.1/NSIF conformance samples
and GDAL's NITF test data, which `tools/fetch_samples.py` downloads. The JITC
files ship reference PNGs, so decoded pixels are compared byte for byte
against renderings this package had no hand in producing. Currently 33 image
segments match their reference exactly, with no mismatches; the rest of the
corpus is checked for self-consistency (windowed and band-subset reads must
agree with a full read) because it has no reference to compare against. Two
files are expected to be refused: a NITF 1.1 file, and a deliberately corrupt
fuzzing input.

The samples are not redistributed here. Both sets are public test data; the
JITC originals are available from NGA/JITC on request.

## Looking at the output

```sh
python tools/gallery.py        # writes gallery/index.html
```

Decodes every segment in `samples/` into a browsable contact sheet: thumbnail,
geometry, compression, interleave, and whether the segment matches its JITC
reference exactly. Pixels the block or pad mask says were never written are
tinted magenta, so masked imagery is distinguishable from merely dark imagery.

Worth running after any change to the decoding path. A transposed block, a
misread band interleave or an off-by-one palette is obvious in a thumbnail and
invisible in a passing assertion.

## Notes on the format

A few things in NITF reliably break readers, and are handled here
deliberately:

- **There is no index.** Segments are located by summing declared subheader
  and data lengths from the end of the file header. One wrong length makes the
  rest of the file unreadable, so lengths are cross-checked: consuming *more*
  than a declared length is fatal, while consuming less is tolerated, because
  real writers leave blank bytes at the end of a subheader and the declared
  length still locates the next structure. Those spare bytes are kept in
  `trailing_bytes`.
- **`ICORDS` changed meaning.** In NITF 2.0 `N` means "no coordinates"; in 2.1
  `N` means UTM north and a blank means none. Getting this wrong shifts the
  remaining 60 bytes of the subheader.
- **Tagged records are parsed leniently**, even in strict mode. A TRE area
  sits beside the pixel data and plays no part in locating it, so one
  malformed record costs the records after it and warns, rather than denying
  you the imagery. The raw bytes stay on the header either way. Writers also
  disagree about whether the 3-byte overflow index is present at all, so it is
  detected rather than assumed.
- **Streaming file headers.** A writer that cannot know the file length writes
  all nines and appends the real header, wrapped in a length-and-delimiter
  envelope, as a `STREAMING_FILE_HEADER` data extension. Without substituting
  it the segment lengths are unusable.
- **Fill bytes.** `0xFF` is legal padding, so a JPEG stream can sit behind a
  run of them; the start marker is located rather than assumed.
- **Sub-byte pixels** are packed continuously across a whole block, MSB first,
  with only the end of the block padded to a byte. Rows are not individually
  aligned.

## Licence

MIT.
