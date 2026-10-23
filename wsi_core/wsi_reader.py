import os
from pathlib import Path

import numpy as np
import openslide
from PIL import Image


class SdpcSlide:
    """OpenSlide-compatible adapter around the legacy Windows SDPC decoder."""

    is_sdpc = True

    def __init__(self, path):
        if os.name != "nt":
            raise RuntimeError("sdpc-win is only supported on Windows")

        previous_cwd = os.getcwd()
        try:
            from sdpc import Sdpc
        finally:
            # sdpc-win changes cwd while loading its bundled DLLs.
            os.chdir(previous_cwd)

        self._slide = Sdpc(str(path))
        self.path = str(path)
        self.level_count = int(self._slide.level_count)
        self.level_dimensions = tuple(tuple(map(int, dim)) for dim in self._slide.level_dimensions)
        self.level_downsamples = tuple(float(value) for value in self._slide.level_downsample)

        header = self._slide.sdpc.contents.picHead.contents
        self.scan_magnification = float(header.rate)
        self.base_mpp = float(header.ruler)
        if not np.isfinite(self.base_mpp) or self.base_mpp <= 0:
            self.close()
            raise ValueError(f"Invalid SDPC level-0 MPP metadata: {self.base_mpp!r}")

    @property
    def dimensions(self):
        return self.level_dimensions[0]

    def get_best_level_for_downsample(self, downsample):
        return min(
            range(self.level_count),
            key=lambda level: abs(self.level_downsamples[level] - float(downsample)),
        )

    def read_region(self, location, level, size):
        level = int(level)
        width, height = map(int, size)
        if not 0 <= level < self.level_count:
            raise ValueError(f"Invalid level {level}; slide has {self.level_count} levels")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid region size: {(width, height)}")

        downsample = self.level_downsamples[level]
        level_x = int(np.floor(float(location[0]) / downsample))
        level_y = int(np.floor(float(location[1]) / downsample))
        level_width, level_height = self.level_dimensions[level]

        src_x0 = max(0, level_x)
        src_y0 = max(0, level_y)
        src_x1 = min(level_width, level_x + width)
        src_y1 = min(level_height, level_y + height)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        if src_x1 > src_x0 and src_y1 > src_y0:
            read_width = src_x1 - src_x0
            read_height = src_y1 - src_y0
            source_location = (
                int(round(src_x0 * downsample)),
                int(round(src_y0 * downsample)),
            )
            patch = self._slide.read_region(
                source_location, level, (read_width, read_height)
            )
            patch = np.asarray(patch, dtype=np.uint8)
            dst_x = src_x0 - level_x
            dst_y = src_y0 - level_y
            canvas[dst_y:dst_y + read_height, dst_x:dst_x + read_width] = patch

        return Image.fromarray(canvas, mode="RGB")

    def close(self):
        slide = getattr(self, "_slide", None)
        if slide is not None:
            slide.close()
            self._slide = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class OpenSlideAdapter:
    """OpenSlide wrapper with MPP metadata and a bounded virtual pyramid.

    Some 3DHISTECH TIFF exports in this cohort contain only level 0. CLAM's
    segmentation step needs a low-resolution overview, while reading the full
    100k-pixel image at once can exhaust RAM. For those files this adapter
    exposes virtual power-of-two levels and renders them in bounded chunks.
    Native pyramidal files continue to use their original levels unchanged.
    """

    is_sdpc = False

    def __init__(self, path, max_source_edge=4096):
        self._slide = openslide.open_slide(str(path))
        self.path = str(path)
        self.properties = self._slide.properties
        self._max_source_edge = int(max_source_edge)

        mpp_x = self.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = self.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        try:
            mpp_x = float(mpp_x)
            mpp_y = float(mpp_y)
        except (TypeError, ValueError):
            self.close()
            raise ValueError(f"Missing or invalid OpenSlide MPP metadata: {mpp_x!r}, {mpp_y!r}")
        if not (np.isfinite(mpp_x) and np.isfinite(mpp_y) and mpp_x > 0 and mpp_y > 0):
            self.close()
            raise ValueError(f"Invalid OpenSlide MPP metadata: {mpp_x!r}, {mpp_y!r}")
        if abs(mpp_x - mpp_y) / max(mpp_x, mpp_y) > 0.05:
            self.close()
            raise ValueError(f"Anisotropic OpenSlide MPP metadata: {mpp_x}, {mpp_y}")
        self.base_mpp = (mpp_x + mpp_y) / 2.0

        native_dimensions = tuple(tuple(map(int, dim)) for dim in self._slide.level_dimensions)
        native_downsamples = tuple(float(value) for value in self._slide.level_downsamples)
        self._virtual = len(native_dimensions) == 1
        if self._virtual:
            width, height = native_dimensions[0]
            downsamples = [1.0]
            while max(width / downsamples[-1], height / downsamples[-1]) > 1024:
                downsamples.append(downsamples[-1] * 2.0)
            self.level_downsamples = tuple(downsamples)
            self.level_dimensions = tuple(
                (int(np.ceil(width / ds)), int(np.ceil(height / ds)))
                for ds in self.level_downsamples
            )
        else:
            self.level_downsamples = native_downsamples
            self.level_dimensions = native_dimensions
        self.level_count = len(self.level_dimensions)

    @property
    def dimensions(self):
        return self.level_dimensions[0]

    def get_best_level_for_downsample(self, downsample):
        return min(
            range(self.level_count),
            key=lambda level: abs(self.level_downsamples[level] - float(downsample)),
        )

    @staticmethod
    def _rgba_on_white(image):
        if image.mode != "RGBA":
            return image.convert("RGB")
        canvas = Image.new("RGB", image.size, "white")
        canvas.paste(image, mask=image.getchannel("A"))
        return canvas

    def read_region(self, location, level, size):
        level = int(level)
        width, height = map(int, size)
        if not 0 <= level < self.level_count:
            raise ValueError(f"Invalid level {level}; slide has {self.level_count} levels")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid region size: {(width, height)}")

        if not self._virtual or level == 0:
            return self._rgba_on_white(self._slide.read_region(location, level, (width, height)))

        downsample = self.level_downsamples[level]
        chunk_edge = max(1, int(self._max_source_edge // downsample))
        output = Image.new("RGB", (width, height), "white")
        for out_y in range(0, height, chunk_edge):
            out_h = min(chunk_edge, height - out_y)
            for out_x in range(0, width, chunk_edge):
                out_w = min(chunk_edge, width - out_x)
                source_x = int(round(float(location[0]) + out_x * downsample))
                source_y = int(round(float(location[1]) + out_y * downsample))
                source_w = max(1, int(round(out_w * downsample)))
                source_h = max(1, int(round(out_h * downsample)))
                tile = self._rgba_on_white(
                    self._slide.read_region(
                        (source_x, source_y), 0, (source_w, source_h)
                    )
                )
                if tile.size != (out_w, out_h):
                    tile = tile.resize((out_w, out_h), Image.Resampling.BILINEAR)
                output.paste(tile, (out_x, out_y))
        return output

    def close(self):
        slide = getattr(self, "_slide", None)
        if slide is not None:
            slide.close()
            self._slide = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def open_wsi(path):
    path = str(Path(path))
    if Path(path).suffix.lower() == ".sdpc":
        return SdpcSlide(path)
    return OpenSlideAdapter(path)
