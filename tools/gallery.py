#!/usr/bin/env python3
"""Render every sample file to a browsable contact sheet.

Decodes each image segment in ``samples/``, writes a thumbnail, and builds an
HTML page showing them next to the segment's geometry and the result of
comparing it with its JITC reference PNG. It is a way to *look* at what the
reader produces: a transposed block, a misread interleave or an off-by-one
palette is obvious in a thumbnail and invisible in a passing assertion.

Pixels whose block was never written, or that carry the pad value, are tinted
magenta so masked imagery can be told apart from merely dark imagery.

    python tools/gallery.py && open gallery/index.html

Needs Pillow.
"""

import argparse
import html
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import vibenitf  # noqa: E402
from vibenitf.errors import NitfError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")

#: Pixels marked invalid by the block or pad mask are painted this colour.
INVALID_TINT = (255, 0, 170)


def viewable(segment):
    """An 8-bit RGB rendering of a segment, stretched to be worth looking at.

    This is deliberately not the reference-matching rendering the tests use:
    here the goal is for a human to see structure, so low-contrast and deep
    pixel types get a percentile stretch they would not get in a comparison.
    """
    palettised = any(t is not None for t in segment.lookup_tables)
    array = segment.read(apply_lut=palettised)
    if array.ndim == 2:
        array = array[:, :, None]

    if array.dtype.kind == "c":
        array = np.abs(array)

    sub = segment.subheader
    if array.shape[2] > 3:
        # Show the bands flagged as red, green and blue; otherwise the first.
        order = {"R": 0, "G": 1, "B": 2}
        wanted = [
            i for i, b in enumerate(sub.bands) if b.irepband.strip().upper() in order
        ]
        if len(wanted) == 3:
            wanted.sort(key=lambda i: order[sub.bands[i].irepband.strip().upper()])
            array = array[:, :, wanted]
        else:
            array = array[:, :, :1]
    elif array.shape[2] == 2:
        array = array[:, :, :1]

    if not palettised and (sub.pvtype == "B" or sub.nbpp == 1):
        eight = (array.astype(np.uint8) * 255)
    elif array.dtype == np.uint8 and sub.abpp in (0, 8, None):
        eight = array
    else:
        values = array.astype(np.float64)
        low, high = np.percentile(values, [2.0, 98.0])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high <= low:
            eight = np.zeros(values.shape, np.uint8)
        else:
            eight = np.clip((values - low) * (255.0 / (high - low)), 0, 255)
            eight = eight.astype(np.uint8)

    if eight.shape[2] == 1:
        eight = np.repeat(eight, 3, axis=2)
    return eight


def tint_invalid(rgb, segment):
    """Paint masked-out pixels, returning the count so it can be reported."""
    try:
        valid = segment.read_mask()
    except NitfError:
        return rgb, 0
    missing = int((~valid).sum())
    if missing:
        rgb = rgb.copy()
        rgb[~valid] = INVALID_TINT
    return rgb, missing


def thumbnail(rgb, box):
    """Fit an image into `box` pixels, keeping tiny test images legible."""
    from PIL import Image

    image = Image.fromarray(rgb, "RGB")
    width, height = image.size
    scale = min(box / float(width), box / float(height))
    if scale < 1:
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(size, Image.LANCZOS)
    # Upscale small images by whole pixels so the structure stays crisp.
    factor = max(1, int(scale))
    if factor > 1:
        return image.resize((width * factor, height * factor), Image.NEAREST)
    return image


def reference_verdict(segment, path):
    """Compare against the JITC reference PNG, if this segment has one."""
    from PIL import Image

    reference = "%s_%d.ref.png" % (path, segment.index)
    if not os.path.exists(reference):
        return None, "no reference"
    try:
        sys.path.insert(0, ROOT)
        from tests.test_corpus import load_reference, render_like_reference

        ours = render_like_reference(segment)
        ref = load_reference(reference)
        if ours.shape[2] == 1 and ref.shape[2] == 3:
            if (ref[:, :, 0] == ref[:, :, 1]).all() and (
                ref[:, :, 1] == ref[:, :, 2]
            ).all():
                ref = ref[:, :, :1]
            else:
                return None, "reference is colour"
        if ours.shape[2] == 3 and ref.shape[2] == 1:
            ref = np.repeat(ref, 3, axis=2)
        if ours.shape[0] > ref.shape[0] or ours.shape[1] > ref.shape[1]:
            return None, "larger than reference"
        ref = ref[: ours.shape[0], : ours.shape[1], :]
        if np.array_equal(ours, ref):
            return True, "matches reference exactly"
        differing = 100.0 * np.count_nonzero(ours != ref) / ours.size
        return False, "differs from reference on %.2f%% of samples" % differing
    except Exception as exc:  # the verdict must never sink the gallery
        return None, "could not compare: %s" % exc


def collect(paths, box, out_dir):
    """Decode everything, writing thumbnails and returning one row per segment."""
    rows = []
    for path in paths:
        name = os.path.basename(path)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with vibenitf.open(path, strict=False) as nitf:
                    title = nitf.header.ftitle
                    if not nitf.images:
                        # Plenty of conformance files exercise CGM graphics or
                        # text and carry no imagery at all; say so rather than
                        # letting it read as a failure to find any.
                        held = [
                            ("%d graphic" % len(nitf.graphics), nitf.graphics),
                            ("%d text" % len(nitf.texts), nitf.texts),
                            ("%d label" % len(nitf.labels), nitf.labels),
                            ("%d data extension" % len(nitf.data_extensions),
                             nitf.data_extensions),
                        ]
                        carries = ", ".join(w for w, items in held if items)
                        rows.append(
                            dict(name=name, title=title, state="empty", image=None,
                                 facts=[("contains", carries or "nothing")],
                                 note="no image segments, by design")
                        )
                    for segment in nitf.images:
                        rows.append(
                            render_segment(segment, path, name, title, box, out_dir)
                        )
        except NitfError as exc:
            rows.append(
                dict(name=name, title="", state="refused", image=None, facts=[],
                     note="%s: %s" % (type(exc).__name__, exc))
            )
        except Exception as exc:
            rows.append(
                dict(name=name, title="", state="refused", image=None, facts=[],
                     note="unexpected %s: %s" % (type(exc).__name__, exc))
            )
    return rows


def render_segment(segment, path, name, title, box, out_dir):
    sub = segment.subheader
    facts = [
        ("size", "%d x %d x %d" % (segment.rows, segment.cols, segment.bands)),
        ("type", "%s %s-bit%s" % (sub.pvtype, sub.nbpp,
                                  "" if sub.abpp == sub.nbpp else " (%s used)" % sub.abpp)),
        ("compression", sub.ic),
        ("interleave", "IMODE %s" % sub.imode),
        ("blocks", "%d x %d of %d x %d" % (sub.nbpc, sub.nbpr,
                                           sub.pixels_per_block_v, sub.pixels_per_block_h)),
        ("representation", sub.irep),
    ]
    row = dict(name=name, title=title, segment=segment.index, facts=facts, image=None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb = viewable(segment)
            rgb, missing = tint_invalid(rgb, segment)
    except NitfError as exc:
        row["state"] = "undecodable"
        row["note"] = str(exc)
        return row
    except Exception as exc:
        row["state"] = "undecodable"
        row["note"] = "unexpected %s: %s" % (type(exc).__name__, exc)
        return row

    if missing:
        facts.append(("masked out", "%d pixels (shown magenta)" % missing))

    filename = "%s.%d.png" % (name, segment.index)
    thumbnail(rgb, box).save(os.path.join(out_dir, filename))
    row["image"] = filename

    exact, note = reference_verdict(segment, path)
    row["state"] = {True: "verified", False: "mismatch", None: "decoded"}[exact]
    row["note"] = note
    return row


STYLE = """
:root {
  --bg: #11131a; --card: #1a1d27; --edge: #2b3040;
  --ink: #e6e8ef; --dim: #9aa1b4;
  --ok: #4ade80; --warn: #fbbf24; --bad: #f87171; --idle: #60a5fa;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 16px 64px; background: var(--bg); color: var(--ink);
       font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
header { max-width: 1400px; margin: 0 auto 28px; }
h1 { margin: 0 0 6px; font-size: 22px; letter-spacing: -0.01em; }
.sub { color: var(--dim); margin: 0 0 18px; }
.tally { display: flex; flex-wrap: wrap; gap: 8px; }
.pill { border: 1px solid var(--edge); border-radius: 999px; padding: 4px 12px;
        color: var(--dim); background: var(--card); font-size: 13px; }
.pill b { color: var(--ink); font-weight: 600; }
.grid { max-width: 1400px; margin: 0 auto; display: grid; gap: 16px;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
.card { background: var(--card); border: 1px solid var(--edge); border-radius: 12px;
        overflow: hidden; display: flex; flex-direction: column; }
.shot { background: #000; display: flex; align-items: center; justify-content: center;
        min-height: 180px; padding: 8px; }
.shot img { max-width: 100%; height: auto; image-rendering: pixelated; display: block; }
.shot.none { color: var(--dim); font-size: 13px; text-align: center; padding: 28px 16px; }
.body { padding: 12px 14px 14px; display: flex; flex-direction: column; gap: 8px; flex: 1; }
.name { font-weight: 600; font-size: 14px; word-break: break-all; }
.title { color: var(--dim); font-size: 12.5px; }
dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 2px 10px;
     font-size: 12.5px; }
dt { color: var(--dim); } dd { margin: 0; font-variant-numeric: tabular-nums; }
.tag { align-self: flex-start; margin-top: auto; border-radius: 6px; padding: 3px 9px;
       font-size: 12px; font-weight: 600; }
.verified { background: rgba(74,222,128,.14); color: var(--ok); }
.decoded  { background: rgba(96,165,250,.14); color: var(--idle); }
.undecodable, .empty { background: rgba(251,191,36,.14); color: var(--warn); }
.mismatch, .refused  { background: rgba(248,113,113,.14); color: var(--bad); }
.note { color: var(--dim); font-size: 12px; }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg: #f6f7f9; --card: #fff; --edge: #e2e5ec; --ink: #16181d; --dim: #646b7a;
    --ok: #15803d; --warn: #a16207; --bad: #b91c1c; --idle: #1d4ed8;
  }
}
"""

LABELS = {
    "verified": "pixel-exact vs reference",
    "decoded": "decoded",
    "undecodable": "no decoder",
    "mismatch": "MISMATCH",
    "refused": "refused",
    "empty": "no imagery",
}


def write_html(rows, out_dir, box):
    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1

    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>vibenitf sample gallery</title><style>%s</style>" % STYLE,
        "<header><h1>vibenitf sample gallery</h1>",
        "<p class='sub'>Every image segment in <code>samples/</code>, decoded and "
        "rendered. Magenta marks pixels the block or pad mask says were never "
        "written.</p><div class='tally'>",
    ]
    for state in ("verified", "decoded", "undecodable", "mismatch", "refused", "empty"):
        if counts.get(state):
            parts.append(
                "<span class='pill'><b>%d</b> %s</span>" % (counts[state], LABELS[state])
            )
    parts.append("</div></header><main class='grid'>")

    for row in rows:
        parts.append("<article class='card'>")
        if row["image"]:
            parts.append(
                "<div class='shot'><img src='%s' alt='%s' loading='lazy'></div>"
                % (html.escape(row["image"]), html.escape(row["name"]))
            )
        else:
            parts.append("<div class='shot none'>not decoded</div>")
        label = row["name"]
        if row.get("segment") is not None:
            label += " &middot; segment %d" % row["segment"]
        parts.append("<div class='body'><div class='name'>%s</div>" % label)
        if row.get("title"):
            parts.append("<div class='title'>%s</div>" % html.escape(row["title"]))
        if row["facts"]:
            parts.append("<dl>")
            for key, value in row["facts"]:
                parts.append(
                    "<dt>%s</dt><dd>%s</dd>" % (html.escape(key), html.escape(str(value)))
                )
            parts.append("</dl>")
        if row.get("note"):
            parts.append("<div class='note'>%s</div>" % html.escape(row["note"]))
        parts.append(
            "<span class='tag %s'>%s</span></div></article>"
            % (row["state"], LABELS[row["state"]])
        )

    parts.append("</main></html>")
    index = os.path.join(out_dir, "index.html")
    with open(index, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    return index, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "gallery"))
    parser.add_argument("--box", type=int, default=320, help="thumbnail size in pixels")
    parser.add_argument("--samples", default=SAMPLES)
    args = parser.parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("the gallery needs Pillow: pip install pillow", file=sys.stderr)
        return 2

    paths = []
    for base, _dirs, names in os.walk(args.samples):
        for name in names:
            if os.path.splitext(name)[1].lower() in (".ntf", ".nsf"):
                paths.append(os.path.join(base, name))
    paths.sort()
    if not paths:
        print("no samples under %s; run tools/fetch_samples.py" % args.samples,
              file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    rows = collect(paths, args.box, args.out)
    index, counts = write_html(rows, args.out, args.box)
    print("%d segments from %d files" % (len(rows), len(paths)))
    for state, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("  %-14s %d" % (LABELS[state], count))
    print("wrote %s" % index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
