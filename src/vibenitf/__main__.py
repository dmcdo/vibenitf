"""Command line inspection of NITF files.

    python -m vibenitf info scene.ntf
    python -m vibenitf info scene.ntf --tres --raw
    python -m vibenitf export scene.ntf out.npy
    python -m vibenitf export scene.ntf out.png --segment 1 --window 0,0,512,512
"""

import argparse
import sys

from . import __version__, codecs, open_nitf
from .errors import NitfError


def _describe_segment(segment, show_tres=False, show_raw=False):
    sub = segment.subheader
    print("  image %d: %d x %d x %d  %s" % (
        segment.index, segment.rows, segment.cols, segment.bands, segment.dtype.str))
    print("    id          %s  %s" % (sub.iid1 or "-", sub.iid2 or ""))
    print("    pixels      PVTYPE=%s NBPP=%s ABPP=%s PJUST=%s IREP=%s ICAT=%s"
          % (sub.pvtype, sub.nbpp, sub.abpp, sub.pjust, sub.irep, sub.icat))
    print("    layout      IMODE=%s blocks=%dx%d of %dx%d  IC=%s%s"
          % (sub.imode, sub.nbpc, sub.nbpr, sub.pixels_per_block_v,
             sub.pixels_per_block_h, sub.ic,
             "" if sub.comrat is None else " COMRAT=" + sub.comrat))
    if not codecs.is_supported(sub.ic):
        print("    warning     IC=%s cannot be decoded by this build" % sub.ic)
    if sub.igeolo:
        print("    corners     %s (ICORDS=%s)" % (sub.igeolo, sub.icords))
    print("    placement   IDLVL=%s IALVL=%s ILOC=%s -> %s IMAG=%s"
          % (sub.idlvl, sub.ialvl, sub.iloc, segment.location, sub.imag))
    bands = ", ".join(
        "%s%s" % (b.irepband or "-", "/LUT" if b.nluts else "") for b in sub.bands)
    if bands:
        print("    bands       %s" % bands)
    mask = segment.mask
    if mask is not None:
        if mask.block_offsets is None:
            where = "no block offsets, so blocks are contiguous"
        else:
            absent = sum(
                1 for i in range(len(mask.block_offsets)) if mask.is_absent(i))
            where = "%d of %d blocks absent" % (absent, len(mask.block_offsets))
        print("    mask        %s, pad code %s%s"
              % (where,
                 "none" if mask.tpxcd is None else mask.tpxcd.hex(),
                 "" if mask.pad_offsets is None else ", pad mask present"))
    for comment in sub.icom:
        print("    comment     %s" % comment)
    if show_tres and len(segment.tres):
        print("    tres        %s" % ", ".join(
            "%s(%d)" % (t.tag, len(t.data)) for t in segment.tres))
        for record in segment.tres:
            if record.fields:
                for key, value in record.fields.items():
                    print("      %s.%-16s %r" % (record.tag, key, value))
    if show_raw:
        for name, value in sub.raw.items():
            print("      raw %-10s %r" % (name, value))


def cmd_info(args):
    with open_nitf(args.path, strict=not args.lenient) as nitf:
        header = nitf.header
        print("%s: %s %s" % (args.path, header.fhdr, header.fver))
        print("  title       %s" % (header.ftitle or "-"))
        print("  originator  %s  station %s  %s"
              % (header.oname or "-", header.ostaid or "-", header.fdt or "-"))
        print("  security    %s  CLEVEL=%s"
              % (header.security.get("clas", "-"), header.clevel))
        print("  length      %d bytes, header %d"
              % (header.file_length, header.header_length))
        if nitf.streamed_header is not None:
            print("  note        length came from a streaming file header")
        print("  segments    %d image, %d graphic, %d text, %d des, %d res"
              % (len(nitf.images), len(nitf.graphics), len(nitf.texts),
                 len(nitf.data_extensions), len(nitf.reserved_extensions)))
        if len(nitf.tres):
            print("  file tres   %s" % ", ".join(nitf.tres.tags))
        for segment in nitf.images:
            _describe_segment(segment, args.tres, args.raw)
        for segment in nitf.texts:
            print("  text %d: %s (%d bytes)"
                  % (segment.index, segment.subheader.txtitl or "-",
                     segment.data_length))
        for segment in nitf.data_extensions:
            print("  des %d: %s version %s (%d bytes)"
                  % (segment.index, segment.subheader.desid,
                     segment.subheader.desver, segment.data_length))
    return 0


def cmd_export(args):
    window = None
    if args.window:
        try:
            window = tuple(int(v) for v in args.window.split(","))
            if len(window) != 4:
                raise ValueError
        except ValueError:
            print("--window wants four integers: row0,col0,row1,col1",
                  file=sys.stderr)
            return 2
    with open_nitf(args.path, strict=not args.lenient) as nitf:
        if not 0 <= args.segment < len(nitf.images):
            print("no image segment %d (the file has %d)"
                  % (args.segment, len(nitf.images)), file=sys.stderr)
            return 2
        segment = nitf.images[args.segment]
        array = segment.read(window=window, apply_lut=args.lut)

    lowered = args.out.lower()
    if lowered.endswith(".npy"):
        import numpy as np

        np.save(args.out, array)
    else:
        try:
            from PIL import Image
        except ImportError:
            print("writing %s needs Pillow; use a .npy output instead" % args.out,
                  file=sys.stderr)
            return 2
        import numpy as np

        scaled = array
        if scaled.dtype != np.uint8:
            # Images deeper than 8 bits have to be scaled to be viewable.
            scaled = scaled.astype(np.float64)
            low, high = float(scaled.min()), float(scaled.max())
            scaled = np.zeros_like(scaled) if high <= low else (
                (scaled - low) * (255.0 / (high - low)))
            scaled = scaled.astype(np.uint8)
        elif segment.subheader.nbpp == 1:
            scaled = scaled * 255
        if scaled.ndim == 3 and scaled.shape[2] not in (3, 4):
            scaled = scaled[:, :, 0]
        Image.fromarray(scaled).save(args.out)
    print("wrote %s %s %s" % (args.out, array.shape, array.dtype))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="vibenitf", description="Inspect and export NITF imagery.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="print a file's structure")
    info.add_argument("path")
    info.add_argument("--tres", action="store_true",
                      help="list tagged record extensions")
    info.add_argument("--raw", action="store_true",
                      help="dump every raw subheader field")
    info.add_argument("--lenient", action="store_true",
                      help="keep going when declared lengths disagree")
    info.set_defaults(func=cmd_info)

    export = sub.add_parser("export", help="write pixels to .npy or an image")
    export.add_argument("path")
    export.add_argument("out", help="output file; .npy keeps the exact values")
    export.add_argument("--segment", type=int, default=0)
    export.add_argument("--window", help="row0,col0,row1,col1")
    export.add_argument("--lut", action="store_true",
                        help="expand palettised values through the lookup table")
    export.add_argument("--lenient", action="store_true")
    export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except NitfError as exc:
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
