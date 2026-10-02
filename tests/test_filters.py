"""Tests for the neighbourhood filtering module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.config import PipelineConfig
from bgremover.filters import (
    LAPLACIAN_4,
    LAPLACIAN_8,
    box_kernel,
    convolve2d,
    correlate2d,
    denoise,
    gaussian_filter,
    gaussian_kernel,
    laplacian_sharpen,
    mean_filter,
    median_filter,
    pad_image,
    unsharp_mask,
)

RADIUS = 6


def _border_slice(width: int, radius: int) -> slice:
    return slice(radius, width - radius)


def _reference_gray() -> np.ndarray:
    """Deterministic smooth gray image with structure and noise."""
    rng = np.random.default_rng(21)
    rows = np.linspace(20, 230, 40)[:, None]
    cols = np.linspace(0, 60, 48)[None, :]
    smooth = rows + cols * np.sin(np.arange(48) / 5.0)[None, :]
    noisy = smooth + rng.normal(0.0, 6.0, smooth.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def test_pad_image_shapes_and_modes(rect_image) -> None:
    """Padding grows the image and honours the three documented modes."""
    out = pad_image(rect_image, 2, "reflect")
    assert out.shape == (132, 132, 3)
    assert np.array_equal(out, pad_image(rect_image, (2, 2), "reflect"))
    edge = pad_image(rect_image, 3, "edge")
    assert np.all(edge[3, 3:-3] == rect_image[0])
    assert np.all(edge[3:-3, 3] == rect_image[:, 0])
    zero = pad_image(rect_image, 3, "zero")
    assert np.all(zero[0] == 0)
    assert np.array_equal(pad_image(rect_image, 0), rect_image)


def test_pad_image_rejects_bad_input(rect_image) -> None:
    """Unknown modes and negative pads are rejected."""
    with pytest.raises(ValueError):
        pad_image(rect_image, 1, "wrap")
    with pytest.raises(ValueError):
        pad_image(rect_image, -1)


def test_correlate_identity_kernel(rect_image) -> None:
    """A delta kernel returns the input unchanged."""
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[1, 1] = 1.0
    out = correlate2d(rect_image, kernel)
    assert out.dtype == np.float32
    assert np.allclose(out, rect_image)


def test_convolve_flips_the_kernel() -> None:
    """Convolution and correlation differ for an asymmetric kernel."""
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 3] = 100
    kernel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                      dtype=np.float32)
    correlated = correlate2d(image, kernel)
    convolved = convolve2d(image, kernel)
    assert correlated[2, 2] == 100.0
    assert convolved[2, 4] == 100.0


def test_convolve_box_on_constant(rect_image) -> None:
    """A box kernel preserves a constant image."""
    flat = np.full((32, 32), 200, dtype=np.uint8)
    out = convolve2d(flat, box_kernel(3))
    assert np.allclose(out, 200.0)


def test_convolution_rejects_bad_kernels(rect_image) -> None:
    """Even, 1-D and empty kernels are rejected."""
    with pytest.raises(ValueError):
        convolve2d(rect_image, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        convolve2d(rect_image, np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        convolve2d(rect_image, np.ones((3, 3), dtype=np.float32), "wrap")


def test_box_kernel_properties() -> None:
    """The box kernel is square and normalised."""
    assert box_kernel(1).shape == (1, 1)
    assert box_kernel(5).sum() == pytest.approx(1.0)
    assert np.all(box_kernel(3) == 1.0 / 9.0)
    with pytest.raises(ValueError):
        box_kernel(4)
    with pytest.raises(ValueError):
        box_kernel(0)


def test_gaussian_kernel_properties() -> None:
    """The Gaussian kernel sums to 1 and is symmetric."""
    kernel = gaussian_kernel(5, 1.2)
    assert kernel.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(kernel, kernel[::-1, :])
    assert np.allclose(kernel, kernel[:, ::-1])
    assert kernel[2, 2] == kernel.max()
    derived = gaussian_kernel(5, None)
    assert derived.sum() == pytest.approx(1.0, abs=1e-6)
    expected_sigma = 0.3 * (2.0 - 1.0) + 0.8
    separable = np.outer(
        np.exp(-(np.arange(5) - 2.0) ** 2 / (2 * expected_sigma ** 2)),
        np.exp(-(np.arange(5) - 2.0) ** 2 / (2 * expected_sigma ** 2)),
    )
    assert np.allclose(derived, separable / separable.sum(), atol=1e-6)


def test_gaussian_kernel_rejects_bad_input() -> None:
    """Even sizes and non-positive sigma are rejected."""
    with pytest.raises(ValueError):
        gaussian_kernel(4)
    with pytest.raises(ValueError):
        gaussian_kernel(5, 0.0)
    with pytest.raises(ValueError):
        gaussian_kernel(5, -1.0)


@pytest.mark.parametrize("size,sigma", [(3, 1.0), (5, 1.2), (7, 2.0)])
def test_gaussian_filter_matches_opencv(size: int, sigma: float) -> None:
    """The separable Gaussian agrees with ``cv2.GaussianBlur`` inside."""
    gray = _reference_gray()
    mine = gaussian_filter(gray, size, sigma)
    reference = cv2.GaussianBlur(gray, (size, size), sigma)
    inner = mine[RADIUS:-RADIUS, RADIUS:-RADIUS].astype(int)
    expected = reference[RADIUS:-RADIUS, RADIUS:-RADIUS].astype(int)
    assert np.max(np.abs(inner - expected)) <= 1


def test_gaussian_filter_color_matches_opencv() -> None:
    """Colour images are filtered channel by channel."""
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(24, 28, 3), dtype=np.uint8)
    mine = gaussian_filter(rgb, 5, 1.2)
    reference = cv2.GaussianBlur(rgb, (5, 5), 1.2)
    assert np.max(
        np.abs(mine[6:-6, 6:-6].astype(int) - reference[6:-6, 6:-6]
               .astype(int))
    ) <= 1


def test_gaussian_filter_rejects_even_size() -> None:
    """Even kernel sizes are rejected."""
    with pytest.raises(ValueError):
        gaussian_filter(np.zeros((8, 8), dtype=np.uint8), 4)


def test_mean_filter_preserves_constant() -> None:
    """Averaging a constant image is a no-op."""
    flat = np.full((20, 20), 90, dtype=np.uint8)
    assert np.all(mean_filter(flat, 5) == 90)


def test_mean_filter_slows_a_step_edge() -> None:
    """A box filter softens an abrupt step."""
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 200
    out = mean_filter(image, 3)
    assert out.dtype == np.uint8
    assert int(out[10, 5]) == 0
    assert 0 < int(out[10, 10]) < 200
    assert int(out[10, 15]) == 200


def test_median_filter_matches_opencv() -> None:
    """The median filter agrees with ``cv2.medianBlur`` inside."""
    rng = np.random.default_rng(13)
    image = rng.integers(0, 256, size=(24, 26), dtype=np.uint8)
    mine = median_filter(image, 3)
    reference = cv2.medianBlur(image, 3)
    inner = mine[4:-4, 4:-4]
    expected = reference[4:-4, 4:-4]
    assert np.array_equal(inner, expected)


def _salt_and_pepper(image: np.ndarray) -> np.ndarray:
    """Add 5 % salt and 5 % pepper noise deterministically."""
    rng = np.random.default_rng(99)
    out = image.copy()
    height, width = image.shape[:2]
    count = int(0.05 * height * width)
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    out[ys, xs] = np.uint8(0)
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    out[ys, xs] = np.uint8(255)
    return out


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return the peak signal-to-noise ratio in dB."""
    mse = float(np.mean((reference.astype(np.float64) -
                         estimate.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return 99.0
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def test_median_filter_removes_salt_and_pepper() -> None:
    """Median filtering improves the PSNR by more than 6 dB."""
    rows = np.linspace(10, 240, 60)[:, None]
    smooth = np.repeat(rows, 60, axis=1)
    clean = np.clip(smooth, 0, 255).astype(np.uint8)
    noisy = _salt_and_pepper(clean)
    filtered = median_filter(noisy, 3)
    assert _psnr(clean, filtered) - _psnr(clean, noisy) > 6.0


def test_median_filter_rejects_bad_window(rect_image) -> None:
    """Even, non-positive and oversized windows are rejected."""
    gray = rect_image[..., 0]
    with pytest.raises(ValueError):
        median_filter(gray, 4)
    with pytest.raises(ValueError):
        median_filter(gray, 0)
    with pytest.raises(ValueError):
        median_filter(gray, 17)


def test_unsharp_mask_increases_gradient_energy() -> None:
    """Sharpening a blurred step restores edge contrast."""
    image = np.zeros((40, 40), dtype=np.uint8)
    image[:, 20:] = 200
    blurred = gaussian_filter(image, 7, 2.0)
    sharpened = unsharp_mask(blurred, 5, 1.2, 1.0)
    assert edge_energy(sharpened) > edge_energy(blurred)
    assert sharpened.dtype == np.uint8


def edge_energy(gray: np.ndarray) -> float:
    """Return the total absolute horizontal gradient of a gray image."""
    return float(np.abs(np.diff(gray.astype(np.int32), axis=1)).sum())


def test_unsharp_mask_zero_amount_is_identity(rect_image) -> None:
    """``amount = 0`` makes ``g = f + 0 = f``."""
    gray = rect_image[..., 0]
    assert np.array_equal(unsharp_mask(gray, 3, 1.0, 0.0), gray)
    with pytest.raises(ValueError):
        unsharp_mask(gray, 3, 1.0, -1.0)


def test_laplacian_sharpen_uses_the_right_kernel() -> None:
    """The 4- and 8-neighbour variants differ."""
    image = np.full((12, 12), 100, dtype=np.uint8)
    out4 = laplacian_sharpen(image, 4)
    out8 = laplacian_sharpen(image, 8)
    assert np.all(out4 == 100)
    assert np.all(out8 == 100)
    image[6, 6] = 130
    assert laplacian_sharpen(image, 4)[6, 6] > 130
    assert laplacian_sharpen(image, 8)[6, 6] > 130
    with pytest.raises(ValueError):
        laplacian_sharpen(image, 6)


def test_laplacian_kernels_sum_to_zero() -> None:
    """Both Laplacian kernels annihilate a constant image."""
    assert LAPLACIAN_4.sum() == 0.0
    assert LAPLACIAN_8.sum() == 0.0
    assert LAPLACIAN_4[1, 1] == -4.0
    assert LAPLACIAN_8[1, 1] == -8.0


def test_denoise_dispatch(rect_image) -> None:
    """``denoise`` follows ``cfg.denoise`` exactly."""
    gray = rect_image[..., 0]
    assert np.array_equal(
        denoise(gray, PipelineConfig(denoise="none")), gray
    )
    assert np.array_equal(
        denoise(gray, PipelineConfig(denoise="median", denoise_ksize=3)),
        median_filter(gray, 3),
    )
    assert np.array_equal(
        denoise(gray, PipelineConfig(denoise="gaussian",
                                     denoise_ksize=5,
                                     gaussian_sigma=1.2)),
        gaussian_filter(gray, 5, 1.2),
    )


def test_denoise_none_returns_a_copy(rect_image) -> None:
    """``"none"`` never aliases the input array."""
    out = denoise(rect_image, PipelineConfig(denoise="none"))
    assert out is not rect_image
    assert np.array_equal(out, rect_image)
