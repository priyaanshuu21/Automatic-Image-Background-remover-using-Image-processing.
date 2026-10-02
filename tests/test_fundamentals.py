"""Tests for the digital image fundamentals module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.fundamentals import (
    bit_depth_table,
    channels_summary,
    describe_image,
    merge_channels,
    requantize,
    resize_max_side,
    spatial_resample,
    split_channels,
    to_gray,
)


def test_describe_image_color(rect_image) -> None:
    """The summary of a colour image reports every channel."""
    info = describe_image(rect_image)
    assert info.height == rect_image.shape[0]
    assert info.width == rect_image.shape[1]
    assert info.channels == 3
    assert info.dtype == "uint8"
    assert info.bit_depth == 8
    assert info.n_pixels == rect_image.shape[0] * rect_image.shape[1]
    assert info.nbytes == rect_image.size
    assert len(info.per_channel_min) == 3
    assert info.per_channel_min == (30.0, 160.0, 60.0)
    assert info.per_channel_max == (225.0, 225.0, 225.0)
    assert all(0.0 <= v <= 255.0 for v in info.per_channel_mean)


def test_describe_image_gray() -> None:
    """A 2-D array is reported as a single-channel image."""
    gray = np.zeros((5, 7), dtype=np.uint8)
    info = describe_image(gray)
    assert info.channels == 1
    assert (info.height, info.width) == (5, 7)
    assert info.per_channel_min == (0.0,)


def test_describe_image_rejects_bad_rank() -> None:
    """A rank-1 array is not a valid image."""
    with pytest.raises(ValueError):
        describe_image(np.zeros(10, dtype=np.uint8))


def test_to_gray_matches_opencv(random_rgb) -> None:
    """The manual BT.601 conversion matches OpenCV within one level."""
    manual = to_gray(random_rgb).astype(np.int32)
    reference = cv2.cvtColor(random_rgb, cv2.COLOR_RGB2GRAY).astype(np.int32)
    assert np.max(np.abs(manual - reference)) <= 1


def test_to_gray_known_values() -> None:
    """Primary colours map to the well-known luma values."""
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255],
          [0, 0, 0]]],
        dtype=np.uint8,
    )
    gray = to_gray(rgb)[0]
    assert list(gray) == [76, 150, 29, 255, 0]


def test_to_gray_rejects_bad_input() -> None:
    """Wrong rank or dtype is reported clearly."""
    with pytest.raises(ValueError):
        to_gray(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        to_gray(np.zeros((4, 4, 3), dtype=np.float32))


def test_split_and_merge_round_trip(rect_image) -> None:
    """Splitting then merging restores the original image."""
    channels = split_channels(rect_image)
    assert len(channels) == 3
    assert all(channel.shape == rect_image.shape[:2] for channel in channels)
    assert np.array_equal(merge_channels(*channels), rect_image)


def test_split_channels_rejects_gray() -> None:
    """A 2-D array has no channel axis to split."""
    with pytest.raises(ValueError):
        split_channels(np.zeros((4, 4), dtype=np.uint8))


def test_merge_channels_validates() -> None:
    """Empty and mismatched channel lists are rejected."""
    with pytest.raises(ValueError):
        merge_channels()
    with pytest.raises(ValueError):
        merge_channels(
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 5), dtype=np.uint8),
        )


@pytest.mark.parametrize("method", ["nearest", "bilinear"])
def test_spatial_resample_shape(circle_image, method) -> None:
    """Halving the size halves both dimensions."""
    small = spatial_resample(circle_image, 0.5, method=method)
    assert small.shape == (circle_image.shape[0] // 2,
                           circle_image.shape[1] // 2, 3)
    big = spatial_resample(circle_image, 2.0, method=method)
    assert big.shape == (circle_image.shape[0] * 2,
                         circle_image.shape[1] * 2, 3)


def test_spatial_resample_gray_keeps_rank() -> None:
    """A 2-D image stays 2-D after resampling."""
    gray = np.arange(36, dtype=np.uint8).reshape(6, 6)
    out = spatial_resample(gray, 0.5, method="bilinear")
    assert out.ndim == 2
    assert out.shape == (3, 3)


def test_bilinear_of_constant_is_constant() -> None:
    """Bilinear interpolation is exact on a constant image."""
    gray = np.full((16, 16), 77, dtype=np.uint8)
    out = spatial_resample(gray, 1.7, method="bilinear")
    assert out.shape == (27, 27)
    assert np.all(out == 77)


def test_spatial_resample_nearest_preserves_values() -> None:
    """Nearest-neighbour resampling only copies existing samples."""
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = spatial_resample(gray, 0.5, method="nearest")
    assert set(np.unique(out)).issubset(set(np.unique(gray)))


def test_spatial_resample_rejects_bad_arguments() -> None:
    """Non-positive factors and unknown methods are rejected."""
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        spatial_resample(gray, 0.0)
    with pytest.raises(ValueError):
        spatial_resample(gray, -1.0)
    with pytest.raises(ValueError):
        spatial_resample(gray, 2.0, method="bicubic")


def test_requantize_limits_levels() -> None:
    """Four gray levels produce at most four distinct values."""
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    out = requantize(gray, 4)
    assert out.dtype == np.uint8
    assert np.unique(out).size <= 4


def test_requantize_identity_at_256() -> None:
    """Keeping 256 levels is (almost) the identity."""
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, size=(16, 24), dtype=np.uint8)
    out = requantize(gray, 256)
    assert int(np.max(np.abs(out.astype(int) - gray.astype(int)))) <= 1


def test_requantize_rejects_bad_levels() -> None:
    """The number of levels must lie in ``[2, 256]``."""
    gray = np.zeros((4, 4), dtype=np.uint8)
    for levels in (1, 0, 257, -3):
        with pytest.raises(ValueError):
            requantize(gray, levels)
    with pytest.raises(ValueError):
        requantize(np.zeros((4, 4, 3), dtype=np.uint8), 4)


def test_resize_max_side_shrinks() -> None:
    """The longest side is capped and the scale is reported."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    out, scale = resize_max_side(img, 200)
    assert out.shape == (150, 200, 3)
    assert scale == pytest.approx(0.5)


def test_resize_max_side_noop() -> None:
    """An image that already fits is returned untouched with scale 1."""
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    out, scale = resize_max_side(img, 800)
    assert out.shape == img.shape
    assert scale == 1.0
    assert out is not img


def test_resize_max_side_rejects_bad_input() -> None:
    """A non-positive limit is rejected."""
    with pytest.raises(ValueError):
        resize_max_side(np.zeros((4, 4), dtype=np.uint8), 0)


def test_resize_max_side_preserves_mean_on_flat(rect_image) -> None:
    """Area-averaged downscaling of a flat image preserves the level."""
    out, _ = resize_max_side(rect_image, 64)
    assert out.shape[0] == 64
    assert out.dtype == np.uint8


def test_bit_depth_table_contains_key_numbers(rect_image) -> None:
    """The teaching report mentions rows, columns and bits per pixel."""
    text = bit_depth_table(rect_image)
    assert "rows" in text
    assert "bits per pixel  : 24" in text
    assert str(rect_image.shape[1]) in text


def test_channels_summary(rect_image) -> None:
    """The one-line summary names the colour model."""
    assert channels_summary(rect_image).startswith(
        f"{rect_image.shape[1]}x{rect_image.shape[0]} RGB uint8"
    )
    gray = np.zeros((4, 8), dtype=np.uint8)
    assert channels_summary(gray).startswith("8x4 gray uint8")
