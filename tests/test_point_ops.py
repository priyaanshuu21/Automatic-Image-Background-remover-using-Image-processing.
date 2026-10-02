"""Tests for the point processing (intensity mapping) operations."""

from __future__ import annotations

import numpy as np
import pytest

from bgremover.point_ops import (
    apply_lut,
    contrast_stretch,
    intensity_slice,
    log_transform,
    negative,
    piecewise_stretch,
    power_law,
    threshold_binary,
)

_LOW_CONTRAST = np.clip(
    np.linspace(90, 160, 64 * 64).reshape(64, 64), 0, 255
).astype(np.uint8)


def test_negative_twice_is_identity(rect_image) -> None:
    """The negative operation is an involution."""
    once = negative(rect_image)
    twice = negative(once)
    assert np.array_equal(twice, rect_image)
    assert not np.array_equal(once, rect_image)


def test_negative_values() -> None:
    """``s = 255 - r`` exactly."""
    gray = np.array([[0, 1, 128, 254, 255]], dtype=np.uint8)
    assert list(negative(gray)[0]) == [255, 254, 127, 1, 0]


def test_contrast_stretch_spans_full_range() -> None:
    """A low-contrast image is stretched to the whole 0..255 range."""
    out = contrast_stretch(_LOW_CONTRAST)
    assert out.min() == 0
    assert out.max() == 255
    assert out.dtype == np.uint8


def test_contrast_stretch_preserves_order() -> None:
    """Stretching is monotone, so the ranking of levels is kept."""
    rng = np.random.default_rng(5)
    gray = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    out = contrast_stretch(gray, 1.0, 99.0)
    assert np.array_equal(np.argsort(gray.ravel()),
                          np.argsort(out.ravel()))


def test_contrast_stretch_constant_image() -> None:
    """A constant image cannot be stretched and is returned unchanged."""
    flat = np.full((4, 4), 42, dtype=np.uint8)
    out = contrast_stretch(flat)
    assert np.array_equal(out, flat)


def test_contrast_stretch_rejects_bad_percentiles() -> None:
    """Invalid percentile pairs are rejected."""
    with pytest.raises(ValueError):
        contrast_stretch(_LOW_CONTRAST, 90.0, 10.0)
    with pytest.raises(ValueError):
        contrast_stretch(_LOW_CONTRAST, -1.0, 50.0)


def test_piecewise_stretch() -> None:
    """Control points are interpolated linearly."""
    gray = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
    out = piecewise_stretch(gray, [(0, 0), (128, 0), (255, 255)])
    assert out.shape == gray.shape
    assert int(out[0, 0]) == 0
    assert int(out[0, 2]) == 0
    assert int(out[0, 4]) == 255
    assert int(out[0, 1]) < int(out[0, 3])


def test_piecewise_stretch_rejects_bad_points() -> None:
    """Control points outside ``[0, 255]`` are rejected."""
    with pytest.raises(ValueError):
        piecewise_stretch(_LOW_CONTRAST, [(-1, 0)])
    with pytest.raises(ValueError):
        piecewise_stretch(_LOW_CONTRAST, [(0, 300)])


def test_threshold_binary_matches_comparison(rect_image) -> None:
    """The mask is exactly ``gray > t``."""
    gray = rect_image[..., 0]
    for level in (0, 50, 200, 254):
        expected = gray > level
        assert np.array_equal(threshold_binary(gray, level), expected)
        assert np.array_equal(
            threshold_binary(gray, level, invert=True), ~expected
        )


def test_threshold_binary_dtype_is_bool() -> None:
    """The returned mask is boolean, not uint8."""
    mask = threshold_binary(_LOW_CONTRAST, 100)
    assert mask.dtype == np.bool_


def test_threshold_binary_rejects_bad_input() -> None:
    """Colour input, bad levels and bad dtype are rejected."""
    with pytest.raises(ValueError):
        threshold_binary(np.zeros((4, 4, 3), dtype=np.uint8), 10)
    with pytest.raises(ValueError):
        threshold_binary(_LOW_CONTRAST, 256)
    with pytest.raises(ValueError):
        threshold_binary(_LOW_CONTRAST.astype(np.float32), 10)


def test_intensity_slice_highlight_mode() -> None:
    """Highlight mode zeroes everything outside the band."""
    gray = np.array([[0, 50, 100, 150, 200, 255]], dtype=np.uint8)
    out = intensity_slice(gray, 100, 150, mode="highlight", value=255)
    assert list(out[0]) == [0, 0, 255, 255, 0, 0]


def test_intensity_slice_preserve_mode() -> None:
    """Preserve mode leaves the other pixels untouched."""
    gray = np.array([[0, 50, 100, 150, 200, 255]], dtype=np.uint8)
    out = intensity_slice(gray, 100, 150, mode="preserve", value=255)
    assert list(out[0]) == [0, 50, 255, 255, 200, 255]


def test_intensity_slice_rejects_bad_input() -> None:
    """Bad mode, inverted band and bad value are rejected."""
    with pytest.raises(ValueError):
        intensity_slice(_LOW_CONTRAST, 10, 20, mode="mask")
    with pytest.raises(ValueError):
        intensity_slice(_LOW_CONTRAST, 200, 100)
    with pytest.raises(ValueError):
        intensity_slice(_LOW_CONTRAST, 10, 20, value=999)


def test_log_transform_endpoints_and_monotonicity() -> None:
    """The log transform fixes 0 and 255 and never decreases."""
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = log_transform(ramp)
    flat = out.ravel()
    assert int(flat[0]) == 0
    assert int(flat[-1]) == 255
    assert np.all(np.diff(flat.astype(np.int32)) >= 0)


def test_log_transform_with_explicit_c() -> None:
    """An explicit constant changes the slope but stays monotone."""
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = log_transform(ramp, c=20.0)
    assert int(out.ravel()[-1]) == pytest.approx(round(20.0 * np.log(256)),
                                                 abs=1)
    assert np.all(np.diff(out.ravel().astype(np.int32)) >= 0)
    with pytest.raises(ValueError):
        log_transform(ramp, c=0.0)


def test_power_law_gamma_one_is_identity(rect_image) -> None:
    """``gamma = 1`` leaves the image unchanged."""
    out = power_law(rect_image, 1.0)
    assert int(np.max(np.abs(out.astype(int) - rect_image.astype(int)))) <= 1


def test_power_law_brightens_and_darkens(rect_image) -> None:
    """Gamma below 1 brightens, gamma above 1 darkens."""
    bright = power_law(rect_image, 0.5)
    dark = power_law(rect_image, 2.0)
    assert bright.mean() > rect_image.mean()
    assert dark.mean() < rect_image.mean()
    assert bright.max() > rect_image.max()
    assert dark.max() < rect_image.max()
    # A saturated input stays saturated under any gamma.
    ramp = np.array([[0, 255]], dtype=np.uint8)
    assert int(power_law(ramp, 0.5)[0, 1]) == 255
    assert int(power_law(ramp, 2.0)[0, 1]) == 255
    assert int(power_law(ramp, 2.0)[0, 0]) == 0


def test_power_law_rejects_bad_gamma() -> None:
    """Non-positive gamma or scale is rejected."""
    with pytest.raises(ValueError):
        power_law(_LOW_CONTRAST, 0.0)
    with pytest.raises(ValueError):
        power_law(_LOW_CONTRAST, 1.0, c=0.0)


def test_apply_lut_round_trip(rect_image) -> None:
    """A LUT that reverses the levels reverses the image."""
    lut = np.arange(255, -1, -1, dtype=np.uint8)
    assert np.array_equal(apply_lut(rect_image, lut), negative(rect_image))


def test_apply_lut_rejects_bad_table() -> None:
    """Wrong LUT length or dtype raises ``ValueError``."""
    with pytest.raises(ValueError):
        apply_lut(_LOW_CONTRAST, np.zeros(255, dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_lut(_LOW_CONTRAST, np.zeros((256, 1), dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_lut(_LOW_CONTRAST, np.zeros(256, dtype=np.float32))
    with pytest.raises(ValueError):
        apply_lut(_LOW_CONTRAST.astype(np.float32), np.zeros(256,
                                                           dtype=np.uint8))


def test_operations_do_not_mutate_input(rect_image) -> None:
    """No point operation may modify its input array in place."""
    reference = rect_image.copy()
    negative(rect_image)
    contrast_stretch(rect_image)
    power_law(rect_image, 2.0)
    log_transform(rect_image)
    piecewise_stretch(rect_image, [(10, 200)])
    assert np.array_equal(rect_image, reference)
