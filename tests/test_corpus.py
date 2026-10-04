"""Tests against the JITC and GDAL sample files, when they are present.

The JITC samples ship with reference PNGs, which makes them the only way to
check decoded pixels against something this package did not produce. Run
``python tools/fetch_samples.py`` to download them; without them these tests
skip rather than fail.

The references are renderings of the display plane, not raw dumps: the canvas
is sized to allow for the ILOC offset but the segment is drawn from the top
left, deeper-than-8-bit values are scaled down, palettes are expanded and
bi-level is stretched to black and white. `render_like_reference` reproduces
those conventions so a comparison is meaningful.
"""

import os
import warnings

import numpy as np
import pytest

import vibenitf
from vibenitf.errors import UnsupportedCompressionError, UnsupportedFeatureError

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
Image = pytest.importorskip("PIL.Image", reason="Pillow is needed to read the references")


def sample_files():
    if not os.path.isdir(SAMPLES):
        return []
    found = set()
    for root, _dirs, names in os.walk(SAMPLES):
        for name in names:
            if os.path.splitext(name)[1].lower() in (".ntf", ".nsf"):
                found.add(os.path.join(root, name))
    return sorted(found)


FILES = sample_files()

#: Files the reader is expected to refuse, with the reason.
EXPECTED_REFUSALS = {
    "U_0002A.NTF": "NITF 1.1 headers are not supported",
    "oss_fuzz_1525.ntf": "deliberately corrupt fuzzing input",
}

pytestmark = pytest.mark.skipif(
    not FILES, reason="no sample files; run tools/fetch_samples.py"
)


def render_like_reference(seg):
    """The segment's pixels, rendered the way the reference PNGs are."""
    sub = seg.subheader
    palettised = any(t is not None for t in seg.lookup_tables)
    a = seg.read(apply_lut=palettised)
    if a.ndim == 2:
        a = a[:, :, None]
    if not palettised and (sub.pvtype == "B" or sub.nbpp == 1):
        return a.astype(np.uint8) * 255
    abpp = sub.abpp or sub.nbpp
    if abpp > 8:
        a = a >> (abpp - 8)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[2] > 3:
        # A renderer shows the three bands flagged as red, green and blue.
        order = {"R": 0, "G": 1, "B": 2}
        wanted = [
            i
            for i, band in enumerate(sub.bands)
            if band.irepband.strip().upper() in order
        ]
        if len(wanted) == 3:
            wanted.sort(key=lambda i: order[sub.bands[i].irepband.strip().upper()])
            a = a[:, :, wanted]
    return a


def load_reference(path):
    with Image.open(path) as im:
        ref = np.asarray(im)
    if ref.ndim == 2:
        ref = ref[:, :, None]
    if ref.shape[2] == 4:
        ref = ref[:, :, :3]
    return ref


@pytest.mark.parametrize("path", FILES, ids=lambda p: os.path.basename(p))
def test_sample_file_reads(path):
    """Every sample either reads, or refuses for a reason we have recorded."""
    name = os.path.basename(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name in EXPECTED_REFUSALS:
            with pytest.raises(vibenitf.NitfError):
                with vibenitf.open(path) as f:
                    for seg in f.images:
                        seg.read()
            return
        with vibenitf.open(path) as f:
            assert f.version in ("2.0", "2.1")
            for seg in f.images:
                try:
                    array = seg.read()
                except (UnsupportedCompressionError, UnsupportedFeatureError) as exc:
                    # An undecodable codec must name itself, not fail obscurely.
                    assert seg.subheader.ic in str(exc)
                    continue
                assert array.shape[:2] == (seg.rows, seg.cols)
                assert array.dtype == seg.dtype


@pytest.mark.parametrize("path", FILES, ids=lambda p: os.path.basename(p))
def test_pixels_match_reference(path):
    """Decoded pixels equal the JITC reference rendering, exactly."""
    if os.path.basename(path) in EXPECTED_REFUSALS:
        pytest.skip("not readable by design")
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with vibenitf.open(path) as f:
            for seg in f.images:
                reference = "%s_%d.ref.png" % (path, seg.index)
                if not os.path.exists(reference):
                    continue
                try:
                    ours = render_like_reference(seg)
                except (UnsupportedCompressionError, UnsupportedFeatureError):
                    continue
                ref = load_reference(reference)

                if ours.shape[2] == 1 and ref.shape[2] == 3:
                    if (ref[:, :, 0] == ref[:, :, 1]).all() and (
                        ref[:, :, 1] == ref[:, :, 2]
                    ).all():
                        ref = ref[:, :, :1]
                    else:
                        pytest.skip("reference is colour but the segment is not")
                if ours.shape[2] == 3 and ref.shape[2] == 1:
                    ref = np.repeat(ref, 3, axis=2)
                if ref.shape[:2] != ours.shape[:2]:
                    if (
                        ours.shape[0] > ref.shape[0]
                        or ours.shape[1] > ref.shape[1]
                    ):
                        pytest.skip("segment is larger than its reference")
                    ref = ref[: ours.shape[0], : ours.shape[1], :]

                np.testing.assert_array_equal(
                    ours, ref, err_msg="segment %d of %s" % (seg.index, path)
                )
                checked += 1
    if not checked:
        pytest.skip("no usable reference for this file")


def test_windows_agree_with_the_full_read():
    """Windowed reads must return exactly what a full read would, per block."""
    tried = 0
    for path in FILES:
        if os.path.basename(path) in EXPECTED_REFUSALS:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with vibenitf.open(path) as f:
                    for seg in f.images:
                        if seg.subheader.block_count < 2 or seg.rows < 4:
                            continue
                        try:
                            whole = seg.read()
                        except (UnsupportedCompressionError, UnsupportedFeatureError):
                            continue
                        bh, bw = seg.block_shape
                        # Straddle a block boundary deliberately.
                        r0, c0 = max(0, bh - 1), max(0, bw - 1)
                        r1 = min(seg.rows, bh + 2)
                        c1 = min(seg.cols, bw + 2)
                        if r1 <= r0 or c1 <= c0:
                            continue
                        part = seg.read(window=(r0, c0, r1, c1))
                        np.testing.assert_array_equal(
                            part, whole[r0:r1, c0:c1], err_msg=path
                        )
                        tried += 1
            except vibenitf.NitfError:
                continue
    assert tried > 0, "no blocked sample images were exercised"


def test_band_subsets_agree_with_the_full_read():
    tried = 0
    for path in FILES:
        if os.path.basename(path) in EXPECTED_REFUSALS:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with vibenitf.open(path) as f:
                    for seg in f.images:
                        if seg.bands < 2:
                            continue
                        try:
                            whole = seg.read()
                        except (UnsupportedCompressionError, UnsupportedFeatureError):
                            continue
                        for band in range(seg.bands):
                            np.testing.assert_array_equal(
                                seg.read(bands=band), whole[:, :, band], err_msg=path
                            )
                        np.testing.assert_array_equal(
                            seg.read(bands=[seg.bands - 1, 0]),
                            whole[:, :, [seg.bands - 1, 0]],
                        )
                        tried += 1
            except vibenitf.NitfError:
                continue
    assert tried > 0, "no multi-band sample images were exercised"
