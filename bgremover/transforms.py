"""Decoupled frequency-transform analysis tools.

These classical signal transforms exist for teaching and for the ``analyze``
command: they inspect an image's spectrum but never feed the segmentation
pipeline, which stays purely spatial.  Only NumPy is used.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dft2",
    "idft2",
    "magnitude_spectrum",
    "dct2",
    "idct2",
    "hadamard_matrix",
    "hadamard_transform",
]


def _as_float_field(image: np.ndarray) -> np.ndarray:
    """Validate a 2-D image and return it as ``float64``."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2-D image, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError("image must not be empty")
    return array.astype(np.float64)


def dft2(gray: np.ndarray) -> np.ndarray:
    """Return the centred-free 2-D discrete Fourier transform.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``complex128`` spectrum of shape ``(H, W)`` with the zero
        frequency at index ``(0, 0)``.

    Raises
    ------
    ValueError
        For a bad image.
    """
    return np.fft.fft2(_as_float_field(gray))


def idft2(spectrum: np.ndarray) -> np.ndarray:
    """Invert :func:`dft2` and return the real part.

    Parameters
    ----------
    spectrum:
        2-D complex spectrum of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``float64`` image of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad spectrum.
    """
    array = np.asarray(spectrum)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"expected a non-empty 2-D spectrum, got shape {array.shape}"
        )
    return np.fft.ifft2(array).real.astype(np.float64)


def magnitude_spectrum(
    spectrum: np.ndarray,
    log_scale: bool = True,
) -> np.ndarray:
    """Return the centred, displayable magnitude spectrum.

    The zero frequency is shifted to the centre with ``fftshift``; with
    ``log_scale`` the ``log1p`` magnitudes are normalised to ``[0, 1]``
    so a figure shows both the DC peak and the faint high frequencies.

    Parameters
    ----------
    spectrum:
        2-D complex spectrum as returned by :func:`dft2`.
    log_scale:
        Apply ``log1p`` and normalise to ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of the same shape.

    Raises
    ------
    ValueError
        For a bad spectrum.
    """
    array = np.asarray(spectrum)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"expected a non-empty 2-D spectrum, got shape {array.shape}"
        )
    shifted = np.fft.fftshift(np.abs(array)).astype(np.float64)
    if not log_scale:
        return shifted
    logged = np.log1p(shifted)
    peak = float(logged.max())
    if peak <= 0.0:
        return np.zeros_like(logged)
    return (logged / peak).astype(np.float64)


def _dct_matrix(size: int) -> np.ndarray:
    """Return the orthonormal DCT-II matrix of the given size."""
    side = int(size)
    if side < 1:
        raise ValueError(f"size must be >= 1, got {size!r}")
    samples = np.arange(side, dtype=np.float64)
    basis = np.arange(side, dtype=np.float64)[:, None]
    matrix = np.cos(
        np.pi * (2.0 * samples + 1.0) * basis / (2.0 * side)
    )
    matrix *= np.sqrt(2.0 / side)
    matrix[0] /= np.sqrt(2.0)
    return matrix


def dct2(gray: np.ndarray) -> np.ndarray:
    """Return the 2-D orthonormal DCT-II of an image.

    The separable formulation ``C @ X @ C.T`` compacts most of the
    signal energy into the top-left (low-frequency) coefficients.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``float64`` coefficient array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad image.
    """
    image = _as_float_field(gray)
    rows = _dct_matrix(image.shape[0])
    cols = _dct_matrix(image.shape[1])
    return (rows @ image @ cols.T).astype(np.float64)


def idct2(coefficients: np.ndarray) -> np.ndarray:
    """Invert :func:`dct2` through the transposed basis.

    Parameters
    ----------
    coefficients:
        2-D DCT coefficient array of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``float64`` image of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad coefficient array.
    """
    array = np.asarray(coefficients, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"expected a non-empty 2-D array, got shape {array.shape}"
        )
    rows = _dct_matrix(array.shape[0])
    cols = _dct_matrix(array.shape[1])
    return (rows.T @ array @ cols).astype(np.float64)


def hadamard_matrix(order: int) -> np.ndarray:
    """Return the Sylvester Hadamard matrix of the given order.

    ``H(1) = [[1]]`` and ``H(2n) = [[H, H], [H, -H]]``; every entry is
    ``+1`` or ``-1`` and distinct rows are orthogonal.

    Parameters
    ----------
    order:
        A power of two, at least 1.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(order, order)``.

    Raises
    ------
    ValueError
        If the order is not a power of two.
    """
    size = int(order)
    if size < 1 or size & (size - 1):
        raise ValueError(
            f"order must be a power of two, got {order!r}"
        )
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < size:
        matrix = np.block(
            [[matrix, matrix], [matrix, -matrix]]
        )
    return matrix


def hadamard_transform(image: np.ndarray) -> np.ndarray:
    """Apply the normalised separable Hadamard transform.

    ``Y = H @ X @ H / N`` is its own inverse, so applying it twice
    returns the input; the DC coefficient ``Y[0, 0]`` holds the image
    sum divided by ``N``.

    Parameters
    ----------
    image:
        2-D square array whose side is a power of two.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of the same shape.

    Raises
    ------
    ValueError
        For a non-square image or a side that is not a power of two.
    """
    field = _as_float_field(image)
    height, width = field.shape
    if height != width:
        raise ValueError(
            f"image must be square, got shape {field.shape}"
        )
    matrix = hadamard_matrix(height)
    return (matrix @ field @ matrix / float(height)).astype(np.float64)
