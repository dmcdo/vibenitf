# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
pip install -e ".[test]"                  # editable install with pytest + Pillow
python -m pytest                          # full suite (corpus tests skip without samples)
python -m pytest tests/test_synthetic.py  # no downloads needed
python -m pytest "tests/test_synthetic.py::test_every_imode_blocked[S]" -q   # one case
python -m pytest tests/test_corpus.py -q -k i_3301k                          # one sample file

python tools/fetch_samples.py             # ~62 MB of public samples into samples/ (gitignored)
python tools/fetch_samples.py gdal        # just one set; re-running only fetches what is missing

python -m vibenitf info samples/jitc21/i_3301k.ntf --tres   # inspect structure
python -m vibenitf export in.ntf out.npy --window 0,0,512,512
```

Pytest config lives in `pyproject.toml` under `[tool.pytest.ini_options]`. There is no
linter or formatter configured; match the surrounding style instead (`%` formatting
throughout, not f-strings; `class Foo(object)`; docstrings that explain *why* a rule
exists rather than restating the code).

## Architecture

The format drives the design: **a NITF file has no index.** The file header declares a
subheader length and a data length for every segment, and segments then follow nose to
tail. Everything is located by summing those lengths, so a single wrong length makes the
rest of the file unreadable. Two consequences shape the whole package:

1. Parsing is strictly sequential and position-dependent (`fields.Cursor`).
2. Length cross-checks are a correctness feature, not a nicety (`headers._finish`).

### Reading pipeline

`file.open_nitf` mmaps the file → `headers.parse_file_header` → `NitfFile._walk` computes
cumulative offsets and constructs one `image.ImageSegment` per image plus `file.Segment`
for every other type → `_collect_tres` and `_resolve_overflow` attach metadata. Nothing
reads pixels until `ImageSegment.read()` is called; segments hold a reference to the whole
mmap and slice it lazily.

`fields.Cursor` is the only thing that touches raw bytes for headers. It consumes
fixed-length ASCII fields in order and records **every** field it reads, under its spec
name, in `.raw` — so no header information is lost even where the package has no opinion
about a field. Parsed values are also attributes under the lowercased NITF name.

### Version differences are isolated, deliberately

`headers.py` handles NITF 2.1 / NSIF 1.0 (identical, `V21`) and NITF 2.0 (`V20`). All
divergence lives in `_security`, `_extension_area`, and version branches inside the
`parse_*_subheader` functions — do not let version checks leak into `image.py` or
`masks.py`. The traps, both already handled:

- **`ICORDS` changed meaning.** In 2.0, `N` means "no coordinates"; in 2.1, `N` means UTM
  north and a *blank* means none. Reading it the wrong way consumes a 60-byte `IGEOLO`
  that is not there and shifts every later field.
- **2.0 has no `FBKGC`, a 27-byte `ONAME`, and label segments (`NUML`)** where 2.1 has a
  reserved `NUMX`.

### The "storage unit" model in `image.py`

This is the central abstraction and the hardest part to re-derive. An image is a grid of
`NBPR` x `NBPC` blocks of `NPPBH` x `NPPBV` pixels, padded at the right and bottom edges.
`IMODE` decides how bands sit inside a block, and a **storage unit** normalises that:

| IMODE | unit | within a unit |
|---|---|---|
| `B` | one whole block, all bands | band planes back to back |
| `P` | one whole block, all bands | values interleaved per pixel |
| `R` | one whole block, all bands | values interleaved per row |
| `S` | one band-plane of one block | the whole image is band-major |

`_unit_index` → `_unit_offset` → `_decode` → `_shape_unit` turns any unit into a uniform
`(block_rows, block_cols, bands)` array, which `read()` then blits into the output. Adding
a mode or pixel layout means touching those four methods and nothing else.

Pixels are **always big-endian** and may be 1, 2, 4 or 12 bits packed continuously across
a whole block, MSB first, with only the end of the block padded to a byte — rows are not
individually aligned. `pixels.py` owns dtype resolution and unpacking.

### Masks, codecs

`masks.py` parses the table that precedes masked imagery (`IC` in `NM`, `M1`, `M3`, `M4`,
`M5`, `M8`). It yields per-unit byte offsets, with `0xFFFFFFFF` meaning the unit was never
written — that sentinel is what makes compressed blocks variable-length and sparse imagery
possible. `_unit_offset` returning `None` propagates as "absent block" all the way to
`read(fill=...)` and `read_mask()`.

`codecs/` is optional by design: the uncompressed path needs only numpy, while JPEG and
JPEG 2000 are delegated to Pillow. The contract for a decoder is a single function,
`decode_block(segment, brow, bcol, band_indices) -> (bh, bw, len(band_indices)) | None`;
`codecs/_support.py` provides the shared block-locating and shape-conforming helpers. New
codecs register in `codecs._decoder`, and anything unimplemented must raise
`UnsupportedCompressionError` naming its `IC` value rather than return wrong pixels. Note
JPEG 2000 usually stores **one codestream for the entire image** regardless of the NITF
block grid, so `j2k.py` decodes the whole segment once and slices blocks out of the cache.

### Strictness policy

Three tiers, applied consistently; preserve this when adding checks:

- **Fatal** — anything that misplaces data: consuming *more* than a declared header
  length, a segment-length sum that disagrees with `FL`, an unreadable required number.
- **Tolerated with a warning** — recoverable slop that costs nothing: consuming *less*
  than a declared length (real writers pad subheaders; the spare bytes are kept in
  `trailing_bytes`), a missing 3-byte TRE overflow index, truncated pixel data when
  `strict=False`.
- **Always lenient, even in strict mode** — tagged records. A TRE area plays no part in
  locating anything, so one malformed record costs the records after it and warns rather
  than denying the caller their imagery. Raw bytes stay on the header regardless.

`strict=False` on `open()` additionally pads truncated blocks with zeros, for damaged
files. `ImageSegment.strict` can be flipped independently of the file's setting.

## Testing strategy

Two complementary layers, and the distinction matters:

- **`tests/test_synthetic.py`** uses `tests/_build.py`, a minimal NITF *writer*, to build
  files in memory where the byte layout is known exactly. This is how every `IMODE`,
  packing width, mask arrangement and both header versions get covered without any
  download. When adding a reader feature, add the writer support here first.
  **Remember the writer must emit big-endian pixels** — a native-order `np.uint16` array
  silently produces a file the reader is right to decode differently.
- **`tests/test_corpus.py`** runs the real JITC and GDAL samples. The JITC files ship
  reference PNGs, which are the only external check on decoded pixels. Those references
  are *display renderings*, not raw dumps: the canvas is sized to allow for the `ILOC`
  offset but the segment is drawn from the top left, deeper-than-8-bit values are scaled
  down by `ABPP - 8`, palettes are expanded and bi-level is stretched to black and white.
  `render_like_reference` reproduces those conventions — a "mismatch" is more often a bug
  in that normalisation than in the reader, so check it before changing `image.py`.
  The corpus layer also asserts self-consistency (windowed and band-subset reads must
  equal the corresponding slice of a full read), which is what caught the band-ordering
  bug that the synthetic tests missed.

`EXPECTED_REFUSALS` in the corpus test lists files the reader is *supposed* to reject (a
NITF 1.1 file, a deliberately corrupt fuzzing input). Samples are not redistributed; they
are gitignored and fetched on demand.

## Public API surface

`vibenitf.open` is an alias for `open_nitf` and shadows the builtin inside the package
namespace — modules that need to open a file use `io.open`. Keep `__init__.py`'s
`__all__` in step with new public names, and keep the supported/unsupported matrix in
`README.md` accurate when codec or version coverage changes.
