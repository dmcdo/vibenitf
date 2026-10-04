"""Decoding an image segment's pixels into a numpy array.

An NITF image is a grid of NBPR x NBPC blocks, each NPPBH x NPPBV pixels, with
the right and bottom edges padded out to fill whole blocks. IMODE decides how
the bands of a block are arranged: all planes of one block together (B), values
interleaved per pixel (P), per row (R), or the whole image band by band (S).
Nothing in the format interleaves at a coarser granularity than this, so one
block is the natural read unit and the only one this module copies.
"""

import math
import warnings

import numpy as np

from . import masks, pixels
from .errors import NitfParseError, UnsupportedFeatureError
from .headers import image_location

#: IMODE values this reader understands.
IMODES = ("B", "P", "R", "S")


class ImageSegment(object):
    """One image segment, and the means to read it as a numpy array.

    Instances hold a reference to the whole file buffer and read lazily; no
    pixels are touched until `read` or `read_block` is called.
    """

    def __init__(self, buf, subheader, data_offset, data_length, index=0, strict=True):
        self.buf = buf
        self.subheader = subheader
        #: Absolute offset of the segment's data, i.e. the mask table when the
        #: image is masked, otherwise the first pixel.
        self.data_offset = data_offset
        self.data_length = data_length
        self.index = index
        self.strict = strict
        self._mask = None
        self._mask_read = False
        self._dtype, self._packed_bits = pixels.resolve_dtype(
            subheader.pvtype, subheader.nbpp
        )
        if subheader.imode not in IMODES and strict:
            raise UnsupportedFeatureError(
                "IMODE=%r is not one of B, P, R, S" % subheader.imode
            )

    # -- geometry ----------------------------------------------------------

    @property
    def rows(self):
        return self.subheader.nrows

    @property
    def cols(self):
        return self.subheader.ncols

    @property
    def bands(self):
        return max(self.subheader.nbands, 1)

    @property
    def dtype(self):
        """Dtype of the array `read` returns, in native byte order."""
        return np.dtype(self._dtype.newbyteorder("="))

    @property
    def shape(self):
        """Shape of the array `read` returns with default arguments."""
        if self.bands == 1:
            return (self.rows, self.cols)
        return (self.rows, self.cols, self.bands)

    @property
    def block_shape(self):
        """(rows, cols) of one block, in pixels."""
        return (self.subheader.pixels_per_block_v, self.subheader.pixels_per_block_h)

    @property
    def location(self):
        """(row, column) placement of this segment from ILOC.

        Segments are positioned relative to the segment named by IALVL, or to
        the common coordinate system when IALVL is zero.
        """
        return image_location(self.subheader.iloc)

    @property
    def lookup_tables(self):
        """Per-band lookup tables as (nluts, nelut) arrays, None where absent."""
        out = []
        for band in self.subheader.bands:
            if not band.nluts:
                out.append(None)
                continue
            table = np.zeros((band.nluts, band.nelut), dtype=np.uint8)
            for i, lut in enumerate(band.luts):
                values = np.frombuffer(lut, dtype=np.uint8)[: band.nelut]
                table[i, : values.size] = values
            out.append(table)
        return out

    @property
    def mask(self):
        """The segment's `BlockMask`, or None when the image is not masked."""
        if not self._mask_read:
            self._mask_read = True
            if self.subheader.is_masked:
                self._mask = masks.parse_block_mask(
                    self.buf, self.data_offset, self.subheader, self.strict
                )
        return self._mask

    @property
    def pixel_offset(self):
        """Absolute offset of the first pixel byte, past any mask table."""
        mask = self.mask
        return self.data_offset + (mask.imdatoff if mask else 0)

    # -- block addressing --------------------------------------------------

    def _check_geometry(self):
        bh, bw = self.block_shape
        if bw <= 0 or bh <= 0:
            raise NitfParseError(
                "block size %dx%d is unusable (NPPBH=%r NPPBV=%r NBPR=%r NBPC=%r)"
                % (
                    bw,
                    bh,
                    self.subheader.nppbh,
                    self.subheader.nppbv,
                    self.subheader.nbpr,
                    self.subheader.nbpc,
                ),
                field="NPPBH",
            )
        return bh, bw

    @property
    def _plane_pixels(self):
        bh, bw = self.block_shape
        return bh * bw

    @property
    def _unit_pixels(self):
        """Pixels in one addressable storage unit.

        For band-sequential images a unit is one band's plane of one block;
        every other mode stores all bands of a block together.
        """
        if self.subheader.imode == "S":
            return self._plane_pixels
        return self._plane_pixels * self.bands

    def _nbytes(self, npixels):
        if self._packed_bits:
            return pixels.packed_size(npixels, self._packed_bits)
        return npixels * self._dtype.itemsize

    def _unit_offset(self, unit_index):
        """Absolute offset of storage unit `unit_index`, or None if absent.

        A mask table gives explicit offsets, which is what lets blocks be
        variable length; without one, units are contiguous and uniform.
        """
        mask = self.mask
        if mask is not None and mask.block_offsets is not None:
            relative = mask.offset_of(unit_index)
            if relative is None:
                return None
            return self.pixel_offset + relative
        return self.pixel_offset + unit_index * self._nbytes(self._unit_pixels)

    def _unit_index(self, brow, bcol, band=0):
        flat = brow * self.subheader.nbpr + bcol
        if self.subheader.imode == "S":
            return band * self.subheader.nbpr * self.subheader.nbpc + flat
        return flat

    # -- decoding ----------------------------------------------------------

    def _raw(self, offset, nbytes, what):
        chunk = self.buf[offset : offset + nbytes]
        if len(chunk) < nbytes:
            if self.strict:
                raise NitfParseError(
                    "file ends %d bytes into the %d bytes of %s"
                    % (len(chunk), nbytes, what),
                    offset=offset,
                )
            warnings.warn(
                "%s is truncated (%d of %d bytes); padding with zeros"
                % (what, len(chunk), nbytes),
                RuntimeWarning,
                stacklevel=3,
            )
            chunk = bytes(chunk) + b"\0" * (nbytes - len(chunk))
        return chunk

    def _decode(self, raw, npixels):
        if self._packed_bits:
            return pixels.unpack(raw, npixels, self._packed_bits).astype(
                self._dtype, copy=False
            )
        return np.frombuffer(raw, dtype=self._dtype, count=npixels)

    def _shape_unit(self, flat, bh, bw, nbands):
        """Reshape a decoded storage unit into (bh, bw, nbands) per IMODE."""
        imode = self.subheader.imode
        if imode == "S" or nbands == 1:
            return flat.reshape(bh, bw, 1)
        if imode == "B":
            return np.moveaxis(flat.reshape(nbands, bh, bw), 0, -1)
        if imode == "P":
            return flat.reshape(bh, bw, nbands)
        if imode == "R":
            return np.moveaxis(flat.reshape(bh, nbands, bw), 1, -1)
        raise UnsupportedFeatureError("IMODE=%r" % imode)

    def _block_bands(self, brow, bcol, band_indices):
        """Decode one block as (bh, bw, len(band_indices)), or None if absent.

        None means the block was never written to the file, which a block mask
        records with an offset of 0xFFFFFFFF.
        """
        bh, bw = self._check_geometry()
        nbands = self.bands
        imode = self.subheader.imode

        if self.subheader.is_compressed:
            from . import codecs

            return codecs.decode_block(self, brow, bcol, band_indices)

        if imode == "S":
            planes = []
            for band in band_indices:
                offset = self._unit_offset(self._unit_index(brow, bcol, band))
                if offset is None:
                    return None
                raw = self._raw(
                    offset,
                    self._nbytes(self._plane_pixels),
                    "block (%d,%d) band %d" % (brow, bcol, band),
                )
                planes.append(self._decode(raw, self._plane_pixels).reshape(bh, bw))
            return np.stack(planes, axis=-1)

        offset = self._unit_offset(self._unit_index(brow, bcol))
        if offset is None:
            return None
        what = "block (%d,%d)" % (brow, bcol)

        if imode == "B" and self._packed_bits is None and len(band_indices) < nbands:
            # Band planes sit back to back inside the block, so a subset can be
            # read without touching the rest. Only safe when pixels are byte
            # aligned, since otherwise a plane need not begin on a byte.
            plane_bytes = self._nbytes(self._plane_pixels)
            planes = []
            for band in band_indices:
                raw = self._raw(offset + band * plane_bytes, plane_bytes, what)
                planes.append(self._decode(raw, self._plane_pixels).reshape(bh, bw))
            return np.stack(planes, axis=-1)

        raw = self._raw(offset, self._nbytes(self._unit_pixels), what)
        block = self._shape_unit(self._decode(raw, self._unit_pixels), bh, bw, nbands)
        if band_indices != list(range(nbands)):
            # Selecting every band in a different order is still a selection.
            block = block[:, :, band_indices]
        return block

    # -- public reading ----------------------------------------------------

    def _resolve_window(self, window):
        if window is None:
            return 0, 0, self.rows, self.cols
        try:
            r0, c0, r1, c1 = window
        except (TypeError, ValueError):
            raise ValueError(
                "window must be (row_start, col_start, row_stop, col_stop)"
            ) from None
        r0, c0 = max(0, int(r0)), max(0, int(c0))
        r1, c1 = min(self.rows, int(r1)), min(self.cols, int(c1))
        if r1 <= r0 or c1 <= c0:
            raise ValueError("window %r is empty or outside the image" % (window,))
        return r0, c0, r1, c1

    def _resolve_bands(self, bands):
        n = self.bands
        if bands is None:
            return list(range(n))
        if isinstance(bands, (int, np.integer)):
            bands = [bands]
        out = []
        for b in bands:
            b = int(b)
            if b < 0:
                b += n
            if not 0 <= b < n:
                raise IndexError("band %r out of range for %d bands" % (b, n))
            out.append(b)
        if not out:
            raise ValueError("no bands selected")
        return out

    def read(
        self,
        window=None,
        bands=None,
        band_axis="last",
        squeeze=True,
        justify=True,
        apply_lut=False,
        fill=0,
    ):
        """Read pixels into a numpy array.

        window
            ``(row_start, col_start, row_stop, col_stop)``, half open and in
            pixels, clipped to the image. Only the blocks it touches are read.
        bands
            A band index, a sequence of indices, or None for every band.
        band_axis
            ``"last"`` for (rows, cols, bands), ``"first"`` for
            (bands, rows, cols).
        squeeze
            Drop the band axis when a single band is returned.
        justify
            Shift left-justified values down to their true magnitude when ABPP
            is narrower than NBPP. See `pixels.apply_justification`.
        apply_lut
            Map values through the segment's lookup tables, expanding a
            palettised image into its component bands.
        fill
            Value for pixels whose block is absent from the file.
        """
        if band_axis not in ("last", "first"):
            raise ValueError("band_axis must be 'last' or 'first'")
        r0, c0, r1, c1 = self._resolve_window(window)
        band_indices = self._resolve_bands(bands)
        bh, bw = self._check_geometry()

        out = np.full((r1 - r0, c1 - c0, len(band_indices)), fill, dtype=self.dtype)
        for brow in range(r0 // bh, math.ceil(r1 / bh)):
            for bcol in range(c0 // bw, math.ceil(c1 / bw)):
                block = self._block_bands(brow, bcol, band_indices)
                if block is None:
                    continue
                top, left = brow * bh, bcol * bw
                sr0, sr1 = max(r0, top), min(r1, top + bh)
                sc0, sc1 = max(c0, left), min(c1, left + bw)
                out[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0, :] = block[
                    sr0 - top : sr1 - top, sc0 - left : sc1 - left, :
                ]

        if justify:
            out = pixels.apply_justification(
                out, self.subheader.abpp, self.subheader.nbpp, self.subheader.pjust
            )
        if apply_lut:
            out = self._apply_lut(out, band_indices)
        if band_axis == "first":
            out = np.moveaxis(out, -1, 0)
            if squeeze and out.shape[0] == 1:
                out = out[0]
        elif squeeze and out.shape[-1] == 1:
            out = out[:, :, 0]
        return out

    def _apply_lut(self, out, band_indices):
        """Expand palettised values through the per-band lookup tables."""
        tables = self.lookup_tables
        expanded = []
        for position, band in enumerate(band_indices):
            table = tables[band] if band < len(tables) else None
            plane = out[:, :, position]
            if table is None:
                expanded.append(plane[:, :, None])
                continue
            index = np.clip(plane.astype(np.intp), 0, table.shape[1] - 1)
            expanded.append(np.moveaxis(table[:, index], 0, -1))
        return np.concatenate(expanded, axis=-1)

    def read_block(self, brow, bcol, bands=None, squeeze=True):
        """Read one whole block, including any edge padding it carries."""
        if not 0 <= brow < self.subheader.nbpc or not 0 <= bcol < self.subheader.nbpr:
            raise IndexError(
                "block (%d,%d) outside the %dx%d block grid"
                % (brow, bcol, self.subheader.nbpc, self.subheader.nbpr)
            )
        band_indices = self._resolve_bands(bands)
        block = self._block_bands(brow, bcol, band_indices)
        if block is None:
            bh, bw = self.block_shape
            block = np.zeros((bh, bw, len(band_indices)), dtype=self.dtype)
        if squeeze and block.shape[-1] == 1:
            return block[:, :, 0]
        return block

    def _block_absent(self, brow, bcol):
        if self.subheader.imode == "S":
            return all(
                self._unit_offset(self._unit_index(brow, bcol, band)) is None
                for band in range(self.bands)
            )
        return self._unit_offset(self._unit_index(brow, bcol)) is None

    def read_mask(self, window=None):
        """Return a bool array that is True where a pixel holds real data.

        Pixels are invalid where their block was never written, and where a
        block flagged in the pad mask carries the pad value TPXCD.
        """
        r0, c0, r1, c1 = self._resolve_window(window)
        valid = np.ones((r1 - r0, c1 - c0), dtype=bool)
        mask = self.mask
        if mask is None:
            return valid
        bh, bw = self._check_geometry()
        pad = mask.pad_value(self._dtype, self.bands)
        band_indices = list(range(self.bands))

        for brow in range(r0 // bh, math.ceil(r1 / bh)):
            for bcol in range(c0 // bw, math.ceil(c1 / bw)):
                top, left = brow * bh, bcol * bw
                sr0, sr1 = max(r0, top), min(r1, top + bh)
                sc0, sc1 = max(c0, left), min(c1, left + bw)
                target = (slice(sr0 - r0, sr1 - r0), slice(sc0 - c0, sc1 - c0))
                if self._block_absent(brow, bcol):
                    valid[target] = False
                    continue
                flagged = any(
                    mask.has_pad(self._unit_index(brow, bcol, band))
                    for band in (band_indices if self.subheader.imode == "S" else [0])
                )
                if pad is None or not flagged:
                    continue
                block = self._block_bands(brow, bcol, band_indices)
                if block is None:
                    valid[target] = False
                    continue
                region = block[sr0 - top : sr1 - top, sc0 - left : sc1 - left, :]
                valid[target] &= ~(region == np.asarray(pad)).all(axis=-1)
        return valid

    def __repr__(self):
        return "<ImageSegment %d %dx%dx%d %s ic=%s imode=%s>" % (
            self.index,
            self.rows,
            self.cols,
            self.bands,
            self.dtype.str,
            self.subheader.ic,
            self.subheader.imode,
        )
