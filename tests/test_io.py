"""Tests for image input and output."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bgremover.io import image_file_info, load_image, save_image


def test_png_round_trip_is_lossless(tmp_path, rect_image) -> None:
    """PNG stores the samples unchanged."""
    path = tmp_path / "rgb.png"
    save_image(path, rect_image)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == rect_image.shape
    assert np.array_equal(loaded, rect_image)


def test_tiff_round_trip_is_lossless(tmp_path, circle_image) -> None:
    """TIFF is a lossless container as well."""
    path = tmp_path / "rgb.tiff"
    save_image(path, circle_image)
    assert np.array_equal(load_image(path), circle_image)


def test_jpeg_round_trip_is_close(tmp_path, circle_image) -> None:
    """JPEG is lossy but must stay within a mean error of 3 levels."""
    path = tmp_path / "rgb.jpg"
    save_image(path, circle_image, jpeg_quality=95)
    loaded = load_image(path)
    assert loaded.shape == circle_image.shape
    error = np.abs(loaded.astype(np.int32) - circle_image.astype(np.int32))
    assert float(error.mean()) < 3.0


def test_rgba_png_round_trip_keeps_alpha(tmp_path, rect_image) -> None:
    """The alpha channel survives a PNG round trip."""
    rgba = np.dstack([rect_image, np.full(rect_image.shape[:2], 128,
                                          dtype=np.uint8)])
    path = tmp_path / "rgba.png"
    save_image(path, rgba)
    with Image.open(path) as handle:
        assert handle.mode == "RGBA"
    raw = np.array(Image.open(path))
    assert np.array_equal(raw, rgba)


def test_rgba_to_jpeg_raises(tmp_path, rect_image) -> None:
    """JPEG has no alpha channel, so writing RGBA there must fail."""
    rgba = np.dstack([rect_image, np.full(rect_image.shape[:2], 200,
                                          dtype=np.uint8)])
    with pytest.raises(ValueError):
        save_image(tmp_path / "bad.jpg", rgba)


def test_missing_file_raises(tmp_path) -> None:
    """Loading a missing file raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_unsupported_extension_raises(tmp_path, rect_image) -> None:
    """Only the documented extensions are accepted."""
    webp = tmp_path / "image.webp"
    webp.write_bytes(b"RIFF")
    with pytest.raises(ValueError):
        load_image(webp)
    with pytest.raises(ValueError):
        save_image(tmp_path / "image.xyz", rect_image)


def test_save_creates_parent_folders(tmp_path, rect_image) -> None:
    """Missing directories are created automatically."""
    path = tmp_path / "a" / "b" / "c" / "rgb.png"
    save_image(path, rect_image)
    assert path.is_file()


def test_gray_round_trip(tmp_path) -> None:
    """A 2-D array is stored as an 8-bit gray image."""
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    path = tmp_path / "gray.png"
    save_image(path, gray)
    with Image.open(path) as handle:
        assert handle.mode == "L"
    assert np.array_equal(load_image(path)[:, :, 0], gray)


def test_bmp_round_trip(tmp_path, rect_image) -> None:
    """BMP is supported for both reading and writing."""
    path = tmp_path / "rgb.bmp"
    save_image(path, rect_image)
    assert np.array_equal(load_image(path), rect_image)


def test_load_gray_file_becomes_rgb(tmp_path, rect_image) -> None:
    """A grayscale file is expanded to three identical channels."""
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    path = tmp_path / "gray.png"
    save_image(path, gray)
    loaded = load_image(path)
    assert loaded.shape == (4, 4, 3)
    assert np.array_equal(loaded[..., 0], gray)
    assert np.array_equal(loaded[..., 0], loaded[..., 2])


def test_load_rgba_file_is_composited_on_white(tmp_path, rect_image) -> None:
    """A fully transparent pixel becomes white after compositing."""
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 0
    path = tmp_path / "alpha.png"
    save_image(path, rgba)
    loaded = load_image(path)
    assert loaded.shape == (4, 4, 3)
    assert np.all(loaded == 255)


def test_load_palette_file_becomes_rgb(tmp_path) -> None:
    """A palette image is converted to RGB."""
    palette = Image.new("P", (4, 4))
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 6))
    path = tmp_path / "palette.png"
    palette.save(path)
    loaded = load_image(path)
    assert loaded.shape == (4, 4, 3)


def test_save_rejects_wrong_dtype(tmp_path, rect_image) -> None:
    """Only ``uint8`` arrays may be written."""
    with pytest.raises(ValueError):
        save_image(tmp_path / "float.png", rect_image.astype(np.float32))


def test_save_rejects_bad_shape(tmp_path) -> None:
    """Arrays with an unsupported channel count are rejected."""
    with pytest.raises(ValueError):
        save_image(tmp_path / "weird.png", np.zeros((4, 4, 5),
                                                    dtype=np.uint8))


def test_image_file_info(tmp_path, rect_image) -> None:
    """File metadata is reported without decoding the pixels."""
    path = tmp_path / "rgb.png"
    save_image(path, rect_image)
    info = image_file_info(path)
    assert info["format"] == "PNG"
    assert info["mode"] == "RGB"
    assert info["width"] == rect_image.shape[1]
    assert info["height"] == rect_image.shape[0]
    assert info["size_bytes"] == path.stat().st_size
    assert info["suffix"] == ".png"


def test_image_file_info_missing(tmp_path) -> None:
    """Metadata of a missing file raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        image_file_info(tmp_path / "missing.png")


def test_corrupt_file_raises(tmp_path) -> None:
    """A file that is not a decodable image raises ``ValueError``."""
    path = tmp_path / "broken.png"
    path.write_bytes(b"definitely not a png")
    with pytest.raises(ValueError):
        load_image(path)
