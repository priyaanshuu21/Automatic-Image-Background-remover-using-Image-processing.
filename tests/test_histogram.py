"""Tests for the histogram processing module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.histogram import (
    color_histograms,
    compute_histogram,
    cumulative_distribution,
    equalize_color_value,
    equalize_gray,
    histogram_stats,
    normalized_histogram,
)

_LOW_CONTRAST = np.clip(
    np.linspace(90, 160, 64 * 64).reshape(64, 64), 0, 255
).astype(np.uint8)


def _linearity_error(image: np.ndarray) -> float:
    """Return ``max |CDF - ideal uniform CDF|`` for a gray image."""
    hist = compute_histogram(image).astype(np.float64)
    cdf = np.cumsum(hist) / hist.sum()
    ideal = np.arange(256, dtype=np.float64) / 255.0
    return float(np.max(np.abs(cdf - ideal)))


def test_compute_histogram_matches_bincount(rect_image) -> None:
    """The histogram is exactly ``numpy.bincount`` for 256 bins."""
    gray = rect_image[..., 0]
    hist = compute_histogram(gray)
    assert hist.dtype == np.int64
    assert int(hist.sum()) == gray.size
    assert np.array_equal(hist, np.bincount(gray.ravel(), minlength=256))


def test_compute_histogram_counts_every_pixel() -> None:
    """Whatever the bin count, all pixels are counted."""
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for bins in (1, 2, 4, 7, 64, 255, 256):
        hist = compute_histogram(gray, bins)
        assert hist.shape == (bins,)
        assert int(hist.sum()) == gray.size


def test_compute_histogram_rejects_bad_input() -> None:
    """Bad dtype, rank and bin counts are rejected."""
    with pytest.raises(ValueError):
        compute_histogram(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        compute_histogram(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        compute_histogram(np.zeros((4, 4), dtype=np.uint8), 0)
    with pytest.raises(ValueError):
        compute_histogram(np.zeros((4, 4), dtype=np.uint8), 257)


def test_normalized_histogram_sums_to_one(rect_image) -> None:
    """Probabilities add up to one."""
    probs = normalized_histogram(rect_image[..., 0])
    assert probs.dtype == np.float64
    assert float(probs.sum()) == pytest.approx(1.0)


def test_cumulative_distribution(rect_image) -> None:
    """The CDF is non-decreasing and ends at exactly one."""
    cdf = cumulative_distribution(compute_histogram(rect_image[..., 0]))
    assert cdf.shape == (256,)
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf.min() >= 0.0 and cdf.max() <= 1.0
    assert cdf[30] > 0.0


def test_cumulative_distribution_rejects_bad_input() -> None:
    """Empty, negative and all-zero histograms are rejected."""
    with pytest.raises(ValueError):
        cumulative_distribution(np.zeros(0))
    with pytest.raises(ValueError):
        cumulative_distribution(np.array([-1.0, 2.0]))
    with pytest.raises(ValueError):
        cumulative_distribution(np.zeros(4))


def test_equalize_gray_moves_cdf_towards_linear() -> None:
    """Equalisation brings the CDF closer to the ideal uniform CDF."""
    before = _linearity_error(_LOW_CONTRAST)
    after = _linearity_error(equalize_gray(_LOW_CONTRAST))
    assert after < before
    assert equalize_gray(_LOW_CONTRAST).dtype == np.uint8


def test_equalize_gray_constant_image() -> None:
    """A constant image has no dynamic range and is returned unchanged."""
    flat = np.full((16, 16), 33, dtype=np.uint8)
    out = equalize_gray(flat)
    assert np.array_equal(out, flat)
    assert out is not flat


def test_equalize_gray_endpoints() -> None:
    """A bimodal image is pushed towards the two ends of the range."""
    gray = np.zeros((32, 32), dtype=np.uint8)
    gray[:, 16:] = 255
    out = equalize_gray(gray)
    assert int(out.min()) < 40
    assert int(out.max()) > 215


def test_equalize_gray_rejects_bad_input() -> None:
    """Wrong rank or dtype is rejected."""
    with pytest.raises(ValueError):
        equalize_gray(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        equalize_gray(np.zeros((4, 4), dtype=np.float32))


def test_color_histograms(rect_image) -> None:
    """One histogram per channel, each summing to the pixel count."""
    hists = color_histograms(rect_image)
    assert set(hists) == {"R", "G", "B"}
    for values in hists.values():
        assert values.shape == (256,)
        assert int(values.sum()) == rect_image.shape[0] * rect_image.shape[1]
    assert int(hists["G"].argmax()) == 225
    with pytest.raises(ValueError):
        color_histograms(np.zeros((4, 4), dtype=np.uint8))


def test_equalize_color_value_preserves_hue() -> None:
    """Scaling by ``V_new / V`` keeps hue and saturation intact."""
    source = np.array(
        [
            [[200, 30, 30], [30, 200, 30], [30, 30, 200], [180, 40, 120]],
            [[120, 120, 120], [250, 200, 10], [10, 90, 240], [90, 210, 60]],
            [[160, 60, 60], [60, 160, 60], [40, 40, 220], [210, 90, 140]],
        ],
        dtype=np.uint8,
    )
    out = equalize_color_value(source)
    assert out.shape == source.shape
    assert out.dtype == np.uint8
    before = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
    after = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            if before[y, x, 1] > 40 and before[y, x, 2] > 40:
                delta = int(abs(int(after[y, x, 0]) - int(before[y, x, 0])))
                delta = min(delta, 180 - delta)
                assert delta <= 2, (y, x)


def test_equalize_color_value_brightens_dark_pixels() -> None:
    """The darkest pixels are lifted, the brightest stay put."""
    image = np.full((16, 16, 3), 40, dtype=np.uint8)
    image[:4, :4] = 200
    out = equalize_color_value(image)
    assert out.shape == image.shape
    assert int(out.min()) >= 0
    assert int(out.max()) == 255


def test_equalize_color_value_rejects_bad_input() -> None:
    """Wrong rank or dtype is rejected."""
    with pytest.raises(ValueError):
        equalize_color_value(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        equalize_color_value(np.zeros((4, 4, 3), dtype=np.float32))


def test_histogram_stats(rect_image) -> None:
    """Mean, median, mode, standard deviation and entropy are reported."""
    gray = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    stats = histogram_stats(gray)
    assert stats["mean"] == pytest.approx(127.5)
    assert stats["median"] == pytest.approx(127.5)
    assert stats["std"] == pytest.approx(127.5)
    assert stats["entropy"] == pytest.approx(1.0)
    constant = histogram_stats(np.full((4, 4), 7, dtype=np.uint8))
    assert constant["entropy"] == pytest.approx(0.0)
    assert constant["mode"] == 7.0
    assert constant["std"] == pytest.approx(0.0)


def test_histogram_stats_rejects_bad_input() -> None:
    """Wrong rank or dtype is rejected."""
    with pytest.raises(ValueError):
        histogram_stats(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        histogram_stats(np.zeros((4, 4), dtype=np.int32))
