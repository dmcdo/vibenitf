"""Reader tests against files built in memory, where the truth is known."""

import numpy as np
import pytest

import vibenitf
from vibenitf.errors import (
    NitfParseError,
    UnsupportedCompressionError,
    UnsupportedFeatureError,
)

from . import _build


def read_one(data, **kwargs):
    with vibenitf.open(data) as f:
        assert len(f.images) == 1
        return f.images[0].read(**kwargs)


# -- basic geometry --------------------------------------------------------


def test_single_band_roundtrip():
    want = (np.arange(12 * 7) % 251).reshape(12, 7).astype(np.uint8)
    got = read_one(_build.simple_image(want))
    assert got.shape == (12, 7)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, want)


def test_header_fields_are_exposed():
    want = np.zeros((4, 4), np.uint8)
    with vibenitf.open(_build.simple_image(want)) as f:
        assert f.version == "2.1"
        assert f.header.ftitle == "test"
        assert f.header.security["clas"] == "U"
        sub = f.images[0].subheader
        assert sub.nrows == 4 and sub.ncols == 4
        assert sub.ic == "NC"
        # Every consumed field is kept verbatim, under its spec name.
        assert sub.raw["NROWS"] == "00000004"
        assert sub.raw["IM"] == "IM"


@pytest.mark.parametrize("imode", ["B", "P", "R", "S"])
def test_every_imode_gives_the_same_pixels(imode):
    want = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    got = read_one(_build.simple_image(want, imode=imode, irep="RGB"))
    assert got.shape == (5, 6, 3)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("imode", ["B", "P", "R", "S"])
def test_every_imode_blocked(imode):
    # 7x9 over a 2x3 grid of 4x4 blocks, so both edges carry pad pixels.
    want = ((np.arange(7 * 9 * 3) * 3) % 256).reshape(7, 9, 3).astype(np.uint8)
    data = _build.simple_image(
        want, imode=imode, irep="RGB", nbpr=3, nbpc=2, nppbh=4, nppbv=4
    )
    np.testing.assert_array_equal(read_one(data), want)


def test_block_padding_is_cropped_not_returned():
    want = np.full((3, 3), 9, np.uint8)
    data = _build.simple_image(want, nbpr=2, nbpc=2, nppbh=2, nppbv=2)
    with vibenitf.open(data) as f:
        seg = f.images[0]
        assert seg.read().shape == (3, 3)
        # The underlying block is still 2x2 and reachable in full.
        assert seg.read_block(1, 1).shape == (2, 2)


# -- pixel types -----------------------------------------------------------


@pytest.mark.parametrize("bits", [1, 2, 4, 12])
def test_sub_byte_and_twelve_bit_packing(bits):
    high = (1 << bits) - 1
    want = (np.arange(6 * 5) % (high + 1)).reshape(6, 5).astype(np.uint16)
    data = _build.simple_image(want.astype(np.uint16), bits=bits, nbpp=bits)
    got = read_one(data)
    np.testing.assert_array_equal(got.astype(np.uint16), want)


def test_bilevel_pixel_type():
    want = np.array([[1, 0, 1, 1, 0], [0, 0, 1, 0, 1]], np.uint8)
    data = _build.simple_image(want, bits=1, nbpp=1, pvtype="B")
    np.testing.assert_array_equal(read_one(data), want)


@pytest.mark.parametrize(
    "pvtype,dtype",
    [("SI", ">i2"), ("R", ">f4"), ("R", ">f8"), ("C", ">c8"), ("INT", ">u4")],
)
def test_signed_float_and_complex(pvtype, dtype):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(0)
    if dtype.kind == "f":
        want = rng.normal(size=(4, 3)).astype(dtype)
    elif dtype.kind == "c":
        want = (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))).astype(dtype)
    else:
        info = np.iinfo(dtype)
        want = rng.integers(info.min, info.max, size=(4, 3)).astype(dtype)
    data = _build.simple_image(want, pvtype=pvtype, nbpp=dtype.itemsize * 8)
    got = read_one(data)
    assert got.dtype == dtype.newbyteorder("=")
    np.testing.assert_array_equal(got, want)


def test_left_justified_values_are_shifted_down():
    # 11 significant bits stored left justified in a 16-bit field.
    values = np.array([[0, 1, 1023, 2047]], np.uint16)
    data = _build.simple_image(
        (values << 5).astype(np.uint16), nbpp=16, abpp=11, pjust="L"
    )
    np.testing.assert_array_equal(read_one(data), values)
    # The raw field is still available when the shift is not wanted.
    np.testing.assert_array_equal(read_one(data, justify=False), values << 5)


# -- windows and band selection -------------------------------------------


def test_window_matches_the_full_read():
    want = ((np.arange(20 * 24) * 7) % 256).reshape(20, 24).astype(np.uint8)
    data = _build.simple_image(want, nbpr=3, nbpc=2, nppbh=8, nppbv=16)
    with vibenitf.open(data) as f:
        seg = f.images[0]
        np.testing.assert_array_equal(seg.read(window=(3, 5, 17, 20)), want[3:17, 5:20])
        np.testing.assert_array_equal(seg.read(window=(0, 0, 1, 1)), want[0:1, 0:1])
        # Windows are clipped to the image rather than erroring.
        np.testing.assert_array_equal(seg.read(window=(10, 10, 999, 999)), want[10:, 10:])


def test_window_outside_the_image_is_rejected():
    data = _build.simple_image(np.zeros((4, 4), np.uint8))
    with vibenitf.open(data) as f:
        with pytest.raises(ValueError):
            f.images[0].read(window=(9, 9, 20, 20))
        with pytest.raises(ValueError):
            f.images[0].read(window=(2, 2, 2, 4))


@pytest.mark.parametrize("imode", ["B", "P", "R", "S"])
def test_band_selection(imode):
    want = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    data = _build.simple_image(want, imode=imode, irep="RGB")
    with vibenitf.open(data) as f:
        seg = f.images[0]
        np.testing.assert_array_equal(seg.read(bands=1), want[:, :, 1])
        np.testing.assert_array_equal(seg.read(bands=[2, 0]), want[:, :, [2, 0]])
        # Every band, reordered: still a selection, not a plain full read.
        np.testing.assert_array_equal(seg.read(bands=[2, 1, 0]), want[:, :, ::-1])
        np.testing.assert_array_equal(seg.read(bands=[0, 0, 0]), want[:, :, [0, 0, 0]])
        np.testing.assert_array_equal(seg.read(bands=-1), want[:, :, 2])
        with pytest.raises(IndexError):
            seg.read(bands=3)


def test_band_axis_first():
    want = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    data = _build.simple_image(want, irep="RGB")
    got = read_one(data, band_axis="first")
    assert got.shape == (3, 2, 3)
    np.testing.assert_array_equal(got, np.moveaxis(want, -1, 0))


def test_squeeze_can_be_turned_off():
    data = _build.simple_image(np.zeros((2, 2), np.uint8))
    assert read_one(data, squeeze=False).shape == (2, 2, 1)


# -- lookup tables ---------------------------------------------------------


def test_lookup_table_expansion():
    # Two palette entries: index 0 is red, index 1 is green.
    lut = np.array([[255, 0], [0, 255], [0, 0]], np.uint8)
    want = np.array([[0, 1], [1, 0]], np.uint8)
    data = _build.simple_image(
        want, bits=1, nbpp=1, irep="RGB/LUT", irepband=["LU"], luts=[lut]
    )
    with vibenitf.open(data) as f:
        seg = f.images[0]
        np.testing.assert_array_equal(seg.read(), want)  # raw indices by default
        colour = seg.read(apply_lut=True)
        assert colour.shape == (2, 2, 3)
        np.testing.assert_array_equal(colour[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(colour[0, 1], [0, 255, 0])


# -- masked imagery --------------------------------------------------------


def masked_file(absent=(), tpxcd=None, pad_flags=()):
    """A 4x4 single-band NM image of 2x2 blocks, with chosen blocks absent."""
    import struct

    want = np.array(
        [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]], np.uint8
    )
    units, _ = _build.encode_pixels(want, "B", 2, 2, 2, 2)
    present = [i for i in range(4) if i not in absent]
    offsets = []
    cursor = 0
    for i in range(4):
        if i in absent:
            offsets.append(0xFFFFFFFF)
        else:
            offsets.append(cursor)
            cursor += 4
    pads = [0 if i in pad_flags else 0xFFFFFFFF for i in range(4)]

    code = b"" if tpxcd is None else bytes([tpxcd])
    table = struct.pack(
        ">IHHH",
        10 + len(code) + 16 + (16 if pad_flags else 0),
        4,
        4 if pad_flags else 0,
        len(code) * 8,
    )
    table += code
    table += b"".join(struct.pack(">I", o) for o in offsets)
    if pad_flags:
        table += b"".join(struct.pack(">I", p) for p in pads)
    pixels = b"".join(units[i].tobytes() for i in present)

    sub = _build.image_subheader_21(
        4, 4, 1, imode="B", nbpr=2, nbpc=2, nppbh=2, nppbv=2, ic="NM"
    )
    return _build.build([(sub, table + pixels)]), want


def test_masked_image_with_every_block_present():
    data, want = masked_file()
    np.testing.assert_array_equal(read_one(data), want)


def test_absent_blocks_read_as_fill_and_are_marked_invalid():
    data, want = masked_file(absent=(1, 2))
    with vibenitf.open(data) as f:
        seg = f.images[0]
        got = seg.read()
        expected = want.copy()
        expected[0:2, 2:4] = 0  # block 1
        expected[2:4, 0:2] = 0  # block 2
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(seg.read(fill=99)[0:2, 2:4], np.full((2, 2), 99))

        valid = seg.read_mask()
        assert not valid[0:2, 2:4].any()
        assert not valid[2:4, 0:2].any()
        assert valid[0:2, 0:2].all()
        assert valid[2:4, 2:4].all()


def test_pad_pixels_are_reported_by_the_mask():
    data, want = masked_file(tpxcd=4, pad_flags=(0,))
    with vibenitf.open(data) as f:
        seg = f.images[0]
        np.testing.assert_array_equal(seg.read(), want)
        valid = seg.read_mask()
        # Value 4 sits in block 0, which is the block flagged as padded.
        assert not valid[1, 1]
        assert valid[0, 0]
        assert valid[2:, 2:].all()


# -- metadata --------------------------------------------------------------


def test_tres_are_parsed_from_both_areas():
    payload = b"01" + b"00000" + b"00000" + b"000" + b"000"
    payload += b" " * (123 - len(payload))
    record = b"BLOCKA" + b"00123" + payload
    assert len(record) == 11 + 123
    ixshd = record
    udhd = b"PIAPRC" + b"00005" + b"hello"
    sub = _build.image_subheader_21(2, 2, 1, ixshd=ixshd)
    data = _build.build([(sub, b"\0" * 4)], udhd=udhd)
    with vibenitf.open(data) as f:
        assert f.tres.tags == ["PIAPRC"]
        assert f.tres["PIAPRC"].data == b"hello"
        assert f.images[0].tres.tags == ["BLOCKA"]
        blocka = f.images[0].tres["BLOCKA"]
        assert blocka.source == "IXSHD"
        assert blocka.fields["BLOCK_INSTANCE"] == 1
        assert "BLOCKA" in f.images[0].tres
        assert f.images[0].tres.get("NOPE") is None


def test_malformed_tre_warns_and_keeps_the_pixels():
    sub = _build.image_subheader_21(2, 2, 1, ixshd=b"this is not a tagged record at all")
    data = _build.build([(sub, b"\x01\x02\x03\x04")])
    with pytest.warns(RuntimeWarning):
        with vibenitf.open(data) as f:
            np.testing.assert_array_equal(
                f.images[0].read(), np.array([[1, 2], [3, 4]], np.uint8)
            )


def test_image_location_is_decoded():
    sub = _build.image_subheader_21(2, 2, 1, iloc="0010000250")
    data = _build.build([(sub, b"\0" * 4)])
    with vibenitf.open(data) as f:
        assert f.images[0].location == (100, 250)


def test_comments_are_read():
    sub = _build.image_subheader_21(2, 2, 1, icom=["first", "second"])
    data = _build.build([(sub, b"\0" * 4)])
    with vibenitf.open(data) as f:
        assert f.images[0].subheader.icom == ["first", "second"]


# -- error handling --------------------------------------------------------


def test_not_a_nitf_file():
    with pytest.raises(NitfParseError, match="not a NITF file"):
        vibenitf.open(b"PK\x03\x04" + b"\0" * 100)


def test_short_file():
    with pytest.raises(NitfParseError):
        vibenitf.open(b"NITF")


def test_unsupported_version_is_named():
    data = bytearray(_build.simple_image(np.zeros((2, 2), np.uint8)))
    data[4:9] = b"01.10"
    with pytest.raises(UnsupportedFeatureError, match="01.10"):
        vibenitf.open(bytes(data))


def test_truncated_pixel_data_raises_in_strict_mode():
    data = _build.simple_image(np.zeros((8, 8), np.uint8))
    # Truncating invalidates FL, so the file itself has to be opened leniently;
    # the segment is then made strict to show a short read is refused.
    with vibenitf.open(data[:-10], strict=False) as f:
        seg = f.images[0]
        seg.strict = True
        with pytest.raises(NitfParseError, match="file ends"):
            seg.read()


def test_truncated_pixel_data_is_padded_when_lenient():
    want = np.arange(64, dtype=np.uint8).reshape(8, 8)
    data = _build.simple_image(want)
    seg_file = vibenitf.open(data[:-10], strict=False)
    try:
        seg = seg_file.images[0]
        seg.strict = False
        with pytest.warns(RuntimeWarning, match="truncated"):
            got = seg.read()
        np.testing.assert_array_equal(got.ravel()[:54], want.ravel()[:54])
        assert (got.ravel()[54:] == 0).all()
    finally:
        seg_file.close()


def test_declared_length_overrun_is_fatal():
    data = _build.simple_image(np.zeros((4, 4), np.uint8))
    with vibenitf.open(data) as f:
        declared = f.header.image_segments[0].subheader
    # Shrinking LISH below the real subheader size misplaces the pixels, so it
    # has to be an error rather than something to warn about and continue.
    shrunk = data.replace(
        b"%03d%06d" % (1, declared), b"%03d%06d" % (1, declared - 20), 1
    )
    assert shrunk != data
    with pytest.raises(NitfParseError, match="LISH declares"):
        vibenitf.open(shrunk)


def test_trailing_blank_bytes_in_a_subheader_are_tolerated():
    want = np.arange(16, dtype=np.uint8).reshape(4, 4)
    data = _build.simple_image(want)
    with vibenitf.open(data) as f:
        declared = f.header.image_segments[0].subheader
        total = f.header.file_length
        end = f.header.header_length + declared

    # Real writers pad the end of a subheader. Grow LISH and FL to take in
    # three blank bytes the reader has no field for; the declared length still
    # locates the pixels, so nothing should move.
    grown = data[:end] + b"   " + data[end:]
    grown = grown.replace(b"%03d%06d" % (1, declared), b"%03d%06d" % (1, declared + 3), 1)
    grown = grown.replace(b"%012d" % total, b"%012d" % (total + 3), 1)

    with vibenitf.open(grown) as f:
        seg = f.images[0]
        assert seg.subheader.trailing_bytes == b"   "
        np.testing.assert_array_equal(seg.read(), want)


def test_unsupported_compression_names_the_code():
    sub = _build.image_subheader_21(4, 4, 1, ic="C4")
    data = _build.build([(sub, b"\0" * 16)])
    with vibenitf.open(data) as f:
        with pytest.raises(UnsupportedCompressionError, match="C4") as info:
            f.images[0].read()
        assert info.value.code == "C4"
        assert "vector quantisation" in str(info.value)
    assert not vibenitf.codecs.is_supported("C4")
    assert vibenitf.codecs.is_supported("NC")


def test_blank_required_number_is_rejected_then_tolerated():
    data = bytearray(_build.simple_image(np.zeros((4, 4), np.uint8)))
    at = data.index(b"00000004" + b"00000004")
    data[at : at + 8] = b"        "
    with pytest.raises(NitfParseError, match="NROWS"):
        vibenitf.open(bytes(data))
    with vibenitf.open(bytes(data), strict=False) as f:
        assert f.images[0].subheader.nrows is None


# -- convenience API -------------------------------------------------------


def test_imread_and_file_read(tmp_path):
    want = np.arange(9, dtype=np.uint8).reshape(3, 3)
    path = tmp_path / "x.ntf"
    path.write_bytes(_build.simple_image(want))
    np.testing.assert_array_equal(vibenitf.imread(path), want)
    np.testing.assert_array_equal(vibenitf.imread(str(path)), want)
    with vibenitf.open(path) as f:
        np.testing.assert_array_equal(f.read(), want)
        assert repr(f).startswith("<NitfFile 2.1 1 image")
        assert "ImageSegment" in repr(f.images[0])


def test_file_can_be_deleted_after_close(tmp_path):
    path = tmp_path / "y.ntf"
    path.write_bytes(_build.simple_image(np.zeros((2, 2), np.uint8)))
    with vibenitf.open(path) as f:
        f.images[0].read()
    path.unlink()  # the mapping must be released for this to work on Windows
    assert not path.exists()


def test_shape_and_dtype_before_reading():
    data = _build.simple_image(np.zeros((6, 7, 3), np.uint8), irep="RGB")
    with vibenitf.open(data) as f:
        seg = f.images[0]
        assert seg.shape == (6, 7, 3)
        assert seg.dtype == np.uint8
        assert seg.bands == 3
        assert seg.block_shape == (6, 7)
        assert seg.mask is None


# -- NITF 2.0 layout -------------------------------------------------------


def test_nitf_20_roundtrip():
    want = (np.arange(6 * 5) % 199).reshape(6, 5).astype(np.uint8)
    with vibenitf.open(_build.simple_image_20(want)) as f:
        assert f.version == "2.0"
        assert f.header.fver == "02.00"
        assert f.header.oname == "tests"
        assert f.header.fbkgc is None  # 2.0 has no background colour field
        np.testing.assert_array_equal(f.images[0].read(), want)


def test_nitf_20_multiband():
    want = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    for imode in ("B", "P", "R", "S"):
        data = _build.simple_image_20(want, imode=imode, irep="RGB")
        with vibenitf.open(data) as f:
            np.testing.assert_array_equal(f.images[0].read(), want)


def test_icords_n_means_no_coordinates_in_20_but_utm_in_21():
    """The one-character ICORDS field changed meaning between versions.

    Reading it the 2.1 way in a 2.0 file would consume a 60-byte IGEOLO that
    is not there, shifting every field after it.
    """
    want = np.arange(4, dtype=np.uint8).reshape(2, 2)

    # 2.0 with ICORDS=N: no IGEOLO follows.
    with vibenitf.open(_build.simple_image_20(want, icords="N")) as f:
        sub = f.images[0].subheader
        assert sub.icords == "N"
        assert sub.igeolo is None
        np.testing.assert_array_equal(f.images[0].read(), want)

    # 2.0 with real coordinates: IGEOLO is present.
    corners = "325900N0850000E" * 4
    data = _build.simple_image_20(want, icords="G", igeolo=corners)
    with vibenitf.open(data) as f:
        sub = f.images[0].subheader
        assert sub.igeolo == corners
        np.testing.assert_array_equal(f.images[0].read(), want)


def test_icords_blank_means_no_coordinates_in_21():
    data = _build.simple_image(np.arange(4, dtype=np.uint8).reshape(2, 2))
    with vibenitf.open(data) as f:
        sub = f.images[0].subheader
        assert sub.icords == ""
        assert sub.igeolo is None
