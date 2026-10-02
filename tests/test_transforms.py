"""Tests for the decoupled frequency-transform tools."""

from __future__ import annotations

import numpy as np
import pytest

from bgremover.transforms import (
    dct2,
    dft2,
    hadamard_matrix,
    hadamard_transform,
    idct2,
    idft2,
    magnitude_spectrum,
)


def test_dft_roundtrip_and_parseval() -> None:
    """The DFT inverts exactly and preserves energy (Parseval)."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(16, 24)).astype(np.float64)
    spectrum = dft2(image)
    assert spectrum.shape == image.shape
    assert np.allclose(idft2(spectrum), image, atol=1e-9)
    energy_image = float((image ** 2).sum())
    energy_spectrum = float((np.abs(spectrum) ** 2).sum() / image.size)
    assert energy_image == pytest.approx(energy_spectrum, rel=1e-9)
    with pytest.raises(ValueError):
        dft2(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError):
        idft2(np.zeros((0, 0), dtype=np.complex128))


def test_dft_of_constant_is_dc_only() -> None:
    """A flat image concentrates everything in the DC coefficient."""
    image = np.full((8, 12), 7.0)
    spectrum = dft2(image)
    assert spectrum[0, 0] == pytest.approx(7.0 * 8 * 12)
    rest = spectrum.copy()
    rest[0, 0] = 0.0
    assert np.abs(rest).max() < 1e-9


def test_magnitude_spectrum() -> None:
    """The centred spectrum peaks at the middle and spans [0, 1]."""
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(32, 32)).astype(np.float64)
    display = magnitude_spectrum(dft2(image))
    assert display.shape == image.shape
    assert float(display.min()) >= 0.0
    assert float(display.max()) <= 1.0
    assert display[16, 16] == pytest.approx(1.0)
    raw = magnitude_spectrum(dft2(image), log_scale=False)
    assert float(raw[16, 16]) == pytest.approx(float(image.sum()))
    assert np.allclose(
        magnitude_spectrum(np.zeros((4, 4), dtype=np.complex128)),
        np.zeros((4, 4)),
    )
    with pytest.raises(ValueError):
        magnitude_spectrum(np.zeros((4, 4, 3)))


def test_dct_roundtrip_and_energy() -> None:
    """The orthonormal DCT-II inverts exactly and preserves energy."""
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(16, 24)).astype(np.float64)
    coefficients = dct2(image)
    assert coefficients.shape == image.shape
    assert np.allclose(idct2(coefficients), image, atol=1e-8)
    assert float((coefficients ** 2).sum()) == pytest.approx(
        float((image ** 2).sum()), rel=1e-9
    )
    assert coefficients[0, 0] == pytest.approx(
        float(image.sum()) / np.sqrt(image.size)
    )
    with pytest.raises(ValueError):
        dct2(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError):
        idct2(np.zeros((4, 4, 3)))


def test_hadamard_matrix() -> None:
    """Sylvester matrices hold +/-1 with orthogonal rows."""
    assert hadamard_matrix(1).tolist() == [[1.0]]
    assert hadamard_matrix(2).tolist() == [[1.0, 1.0], [1.0, -1.0]]
    fourth = hadamard_matrix(4)
    assert fourth.shape == (4, 4)
    assert set(np.unique(fourth).tolist()) == {-1.0, 1.0}
    assert np.allclose(fourth @ fourth.T, 4.0 * np.eye(4))
    assert np.allclose(
        hadamard_matrix(8) @ hadamard_matrix(8).T, 8.0 * np.eye(8)
    )
    for bad in (0, 3, 6, -4):
        with pytest.raises(ValueError):
            hadamard_matrix(bad)


def test_hadamard_transform_is_self_inverse() -> None:
    """Applying the transform twice returns the input exactly."""
    rng = np.random.default_rng(21)
    image = rng.integers(0, 256, size=(8, 8)).astype(np.float64)
    once = hadamard_transform(image)
    assert once.shape == image.shape
    assert once[0, 0] == pytest.approx(float(image.sum()) / 8.0)
    assert np.allclose(hadamard_transform(once), image, atol=1e-8)
    with pytest.raises(ValueError):
        hadamard_transform(np.zeros((8, 12)))
    with pytest.raises(ValueError):
        hadamard_transform(np.zeros((6, 6)))
