"""Colour spaces, colour slicing and the background colour model.

Two colour spaces drive the segmentation:

* **HSV / HSI** -- hue and saturation describe the colour itself, value (or
  intensity) describes the brightness.  They are the basis of hue-based
  slicing.
* **CIELAB** -- a perceptually near-uniform space, so a plain Euclidean
  distance between two colours is meaningful.  This is the space in which
  the background model and the region growing operate.

The background is assumed to be the region that touches the image border.
Its colour is modelled with a robust median plus median-absolute-deviation
(MAD) trimming, optionally with two clusters found by a deterministic
farthest-point k-means.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from bgremover.config import PipelineConfig

LOGGER = logging.getLogger(__name__)

MAD_SCALE = 1.4826
DISTANCE_NORM = 100.0


def _as_rgb(rgb: np.ndarray) -> np.ndarray:
    """Validate an ``uint8`` or float RGB image in ``[0, 1]``."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) RGB image, got {array.shape}"
        )
    if array.size == 0:
        raise ValueError("image must not be empty")
    if not (array.dtype == np.uint8 or
            np.issubdtype(array.dtype, np.floating)):
        raise ValueError(
            f"expected uint8 or floating input, got {array.dtype}"
        )
    return array


def _as_hsv_float(hsv: np.ndarray) -> np.ndarray:
    """Validate an ``(H, W, 3)`` float HSV image."""
    array = np.asarray(hsv)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) HSV image, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"HSV must be float, got {array.dtype}")
    return array


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV.

    ``V = max(R, G, B)``, ``S = (max - min) / max`` (0 when ``max == 0``)
    and, with ``d = max - min``,

    * ``H = 60 * ((G - B) / d mod 6)`` when the maximum is the red channel,
    * ``H = 60 * ((B - R) / d + 2)`` when the maximum is the green channel,
    * ``H = 60 * ((R - G) / d + 4)`` when the maximum is the blue channel,

    and ``H = 0`` for a gray pixel (``d == 0``).

    Parameters
    ----------
    rgb:
        ``uint8`` array of shape ``(H, W, 3)`` in RGB order, or a float
        image already scaled to ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(H, W, 3)`` with ``H`` in
        ``[0, 360)`` and ``S``, ``V`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input shape or dtype is not supported.
    """
    array = _as_rgb(rgb)
    data = array.astype(np.float32)
    if data.max() > 1.0:
        data = data / 255.0
    red, green, blue = data[..., 0], data[..., 1], data[..., 2]
    value = data.max(axis=2)
    minimum = data.min(axis=2)
    delta = value - minimum
    safe_value = np.maximum(value, 1e-12)
    saturation = np.where(value > 0.0, delta / safe_value, 0.0)
    safe_delta = np.maximum(delta, 1e-12)
    hue_red = ((green - blue) / safe_delta) % 6.0
    hue_green = (blue - red) / safe_delta + 2.0
    hue_blue = (red - green) / safe_delta + 4.0
    hue = np.where(
        red >= value,
        hue_red,
        np.where(green >= value, hue_green, hue_blue),
    )
    hue = np.where(delta > 0.0, hue, 0.0)
    hue = np.mod(hue, 6.0) * 60.0
    saturation = np.where(value > 0.0, saturation, 0.0)
    out = np.stack([hue, saturation, value], axis=2)
    return np.clip(out, 0.0, None).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV back to 8-bit RGB.

    Parameters
    ----------
    hsv:
        ``float32`` array of shape ``(H, W, 3)`` as produced by
        :func:`rgb_to_hsv`.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(H, W, 3)`` in RGB order.

    Raises
    ------
    ValueError
        For a bad shape or a non-floating dtype.
    """
    array = _as_hsv_float(hsv)
    hue_sector = np.mod(array[..., 0] / 60.0, 6.0)
    saturation = np.clip(array[..., 1], 0.0, 1.0)
    value = np.clip(array[..., 2], 0.0, 1.0)
    chroma = value * saturation
    second = chroma * (1.0 - np.abs(np.mod(hue_sector, 2.0) - 1.0))
    match = value - chroma
    index = hue_sector.astype(np.int32) % 6
    red = np.select(
        [index == 0, index == 1, index == 2, index == 3, index == 4],
        [chroma, second, np.zeros_like(chroma), np.zeros_like(chroma),
         second],
        default=chroma,
    )
    green = np.select(
        [index == 0, index == 1, index == 2, index == 3, index == 4],
        [second, chroma, chroma, second, np.zeros_like(chroma)],
        default=np.zeros_like(chroma),
    )
    blue = np.select(
        [index == 0, index == 1, index == 2, index == 3, index == 4],
        [np.zeros_like(chroma), np.zeros_like(chroma), second, chroma,
         chroma],
        default=second,
    )
    out = np.stack([red, green, blue], axis=2) + match[..., None]
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def hsv_to_opencv_scale(hsv: np.ndarray) -> np.ndarray:
    """Scale an HSV image to the 8-bit layout used by OpenCV.

    OpenCV stores hue in ``[0, 180)`` for 8-bit images, i.e. half the
    angular range, while saturation and value use the full ``[0, 255]``.

    Parameters
    ----------
    hsv:
        ``float32`` HSV image with ``H`` in ``[0, 360)``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array with ``H / 2``, ``S * 255`` and ``V * 255``.

    Raises
    ------
    ValueError
        For a bad shape or a non-floating dtype.
    """
    array = _as_hsv_float(hsv)
    scaled = np.stack(
        [array[..., 0] / 2.0, array[..., 1] * 255.0, array[..., 2] * 255.0],
        axis=2,
    )
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def hsv_channel_images(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return the three HSV channels as displayable ``uint8`` images.

    Parameters
    ----------
    rgb:
        ``uint8`` RGB image.

    Returns
    -------
    dict
        Keys ``"H"`` (hue scaled to ``0..255``), ``"S"`` and ``"V"``.

    Raises
    ------
    ValueError
        If the input image is invalid.
    """
    hsv = rgb_to_hsv(rgb)
    hue = np.rint(hsv[..., 0] / 360.0 * 255.0)
    return {
        "H": np.clip(hue, 0, 255).astype(np.uint8),
        "S": np.clip(np.rint(hsv[..., 1] * 255.0), 0, 255).astype(np.uint8),
        "V": np.clip(np.rint(hsv[..., 2] * 255.0), 0, 255).astype(np.uint8),
    }


def rgb_to_hsi(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSI (hue, saturation, intensity).

    With ``R``, ``G``, ``B`` normalised to ``[0, 1]``:

    * ``I = (R + G + B) / 3``,
    * ``S = 1 - 3 * min(R, G, B) / (R + G + B)`` (0 when the sum is 0),
    * ``theta = arccos(0.5 * ((R - G) + (R - B)) /
      sqrt((R - G)^2 + (R - B)(G - B)))`` and ``H = theta`` in degrees when
      ``B <= G``, ``H = 360 - theta`` otherwise.

    The arccos argument is clipped to ``[-1, 1]`` and gray pixels, where the
    denominator vanishes, get ``H = 0`` and ``S = 0``.

    Parameters
    ----------
    rgb:
        ``uint8`` RGB image or a float image scaled to ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(H, W, 3)`` with ``H`` in
        ``[0, 360)`` and ``S``, ``I`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input image is invalid.
    """
    array = _as_rgb(rgb)
    data = array.astype(np.float32)
    if data.max() > 1.0:
        data = data / 255.0
    red, green, blue = data[..., 0], data[..., 1], data[..., 2]
    total = red + green + blue
    intensity = total / 3.0
    safe_total = np.maximum(total, 1e-12)
    saturation = 1.0 - 3.0 * np.minimum(np.minimum(red, green),
                                        blue) / safe_total
    diff_rg = red - green
    diff_rb = red - blue
    denominator = np.sqrt(
        np.maximum(diff_rg ** 2 + diff_rb * (green - blue), 0.0)
    )
    numerator = 0.5 * (diff_rg + diff_rb)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(
            denominator > 1e-12, numerator / np.maximum(denominator, 1e-12),
            1.0,
        )
    theta = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    hue = np.where(blue <= green, theta, 360.0 - theta)
    degenerate = denominator <= 1e-12
    hue = np.where(degenerate, 0.0, hue)
    saturation = np.where(total > 0.0, saturation, 0.0)
    saturation = np.clip(saturation, 0.0, 1.0)
    out = np.stack([hue, saturation, intensity], axis=2)
    return np.clip(out, 0.0, None).astype(np.float32)


_SRGB_FROM_LINEAR = (
    (3.2406, -1.5372, -0.4986),
    (-0.9689, 1.8758, 0.0415),
    (0.0557, -0.2040, 1.0570),
)
_RGB_TO_XYZ = (
    (0.4124, 0.3576, 0.1805),
    (0.2126, 0.7152, 0.0722),
    (0.0193, 0.1192, 0.9505),
)
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
_LAB_EPS = (6.0 / 29.0) ** 3
_LAB_KAPPA = 3.0 * (6.0 / 29.0) ** 2
_XYZ_TO_RGB = np.linalg.inv(
    np.array(_RGB_TO_XYZ, dtype=np.float64)
).T


def _srgb_to_linear(data: np.ndarray) -> np.ndarray:
    """Apply the sRGB inverse gamma to values in ``[0, 1]``."""
    return np.where(
        data <= 0.04045,
        data / 12.92,
        np.power((data + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _lab_f(t: np.ndarray) -> np.ndarray:
    """Companded ``f(t)`` of the CIELAB definition."""
    return np.where(
        t > _LAB_EPS,
        np.cbrt(t),
        t / _LAB_KAPPA + 4.0 / 29.0,
    ).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to CIELAB (D65 white point).

    The conversion follows the standard chain: sRGB gamma decoding to linear
    RGB, a linear transformation to CIE XYZ with the sRGB matrix, and the
    CIELAB companding ``f(t) = t^(1/3)`` for ``t > (6/29)^3`` and
    ``t / (3 (6/29)^2) + 4/29`` below it.

    Parameters
    ----------
    rgb:
        ``uint8`` RGB image or a float image scaled to ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(H, W, 3)``; ``L`` in ``[0, 100]``,
        ``a`` and ``b`` roughly in ``[-128, 127]``.

    Raises
    ------
    ValueError
        If the input image is invalid.
    """
    array = _as_rgb(rgb)
    data = array.astype(np.float32)
    if data.max() > 1.0:
        data = data / 255.0
    linear = _srgb_to_linear(data)
    matrix = np.array(_RGB_TO_XYZ, dtype=np.float32)
    xyz = linear.reshape(-1, 3) @ matrix.T
    xyz = xyz.reshape(linear.shape)
    scaled = xyz / _WHITE_D65
    fx, fy, fz = _lab_f(scaled[..., 0]), _lab_f(scaled[..., 1]), \
        _lab_f(scaled[..., 2])
    lightness = np.clip(116.0 * fy - 16.0, 0.0, 100.0)
    channel_a = 500.0 * (fx - fy)
    channel_b = 200.0 * (fy - fz)
    # a and b are signed quantities and must keep their sign.
    channel_a = np.clip(channel_a, -128.0, 127.0)
    channel_b = np.clip(channel_b, -128.0, 127.0)
    return np.stack(
        [lightness, channel_a, channel_b], axis=2
    ).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB back to 8-bit RGB.

    Parameters
    ----------
    lab:
        ``float32`` array of shape ``(H, W, 3)`` as produced by
        :func:`rgb_to_lab`.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB image.

    Raises
    ------
    ValueError
        For a bad shape or a non-floating dtype.
    """
    array = np.asarray(lab)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) Lab image, got {array.shape}"
        )
    data = array.astype(np.float64)
    fy = (data[..., 0] + 16.0) / 116.0
    fx = fy + data[..., 1] / 500.0
    fz = fy - data[..., 2] / 200.0
    fx3, fy3, fz3 = fx ** 3, fy ** 3, fz ** 3
    x = np.where(fx3 > _LAB_EPS, fx3,
                 _LAB_KAPPA * (fx - 4.0 / 29.0))
    y = np.where(fy3 > _LAB_EPS, fy3,
                 _LAB_KAPPA * (fy - 4.0 / 29.0))
    z = np.where(fz3 > _LAB_EPS, fz3,
                 _LAB_KAPPA * (fz - 4.0 / 29.0))
    xyz = np.stack(
        [x * _WHITE_D65[0], y * _WHITE_D65[1], z * _WHITE_D65[2]], axis=2
    )
    inverse = _XYZ_TO_RGB
    linear = xyz.reshape(-1, 3) @ inverse
    linear = np.clip(linear.reshape(xyz.shape), 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def euclidean_color_distance(
    image_a: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """Return the per-pixel Euclidean distance to a reference colour.

    For Lab input this is the CIE76 colour difference ``Delta E``.

    Parameters
    ----------
    image_a:
        ``(H, W, 3)`` colour image.
    ref:
        Either a 3-vector with the reference colour or an image of the same
        shape as ``image_a`` giving a per-pixel reference.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(H, W)`` with the distances.

    Raises
    ------
    ValueError
        For mismatched shapes or a non-3-channel image.
    """
    array = np.asarray(image_a, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) image, got {array.shape}"
        )
    reference = np.asarray(ref, dtype=np.float32)
    if reference.ndim == 1:
        if reference.shape[0] != 3:
            raise ValueError(
                f"reference vector must have 3 entries, got "
                f"{reference.shape[0]}"
            )
        difference = array - reference.reshape(1, 1, 3)
    elif reference.shape == array.shape:
        difference = array - reference
    else:
        raise ValueError(
            f"reference shape {reference.shape} does not match the image "
            f"shape {array.shape}"
        )
    return np.sqrt(
        np.sum(difference ** 2, axis=2)
    ).astype(np.float32)


def _band(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return the inclusive band mask, allowing a wrap-around for hue."""
    if low <= high:
        return (values >= low) & (values <= high)
    return (values >= low) | (values <= high)


def hsv_slice(
    hsv: np.ndarray,
    h_range: tuple[float, float],
    s_range: tuple[float, float],
    v_range: tuple[float, float],
) -> np.ndarray:
    """Select the pixels whose hue, saturation and value fall in a box.

    Parameters
    ----------
    hsv:
        ``float32`` HSV image with ``H`` in ``[0, 360)``, ``S`` and ``V`` in
        ``[0, 1]``.
    h_range:
        Inclusive hue range in degrees; a range such as ``(350, 10)``
        wraps around 0 degrees.
    s_range, v_range:
        Inclusive saturation and value ranges in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``bool`` array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-wrapping reversed range or values outside their domain.
    """
    array = _as_hsv_float(hsv)
    hue_low, hue_high = float(h_range[0]), float(h_range[1])
    sat_low, sat_high = float(s_range[0]), float(s_range[1])
    val_low, val_high = float(v_range[0]), float(v_range[1])
    if not 0.0 <= hue_low <= 360.0 or not 0.0 <= hue_high <= 360.0:
        raise ValueError(
            f"hue range must lie in [0, 360], got {h_range!r}"
        )
    if hue_low > hue_high and hue_high != 0.0 and hue_low != 360.0:
        LOGGER.debug("hue range %s wraps around 0 degrees", h_range)
    for name, low, high in (("saturation", sat_low, sat_high),
                            ("value", val_low, val_high)):
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"{name} range must satisfy 0 <= low <= high <= 1, got "
                f"({low}, {high})"
            )
    hue_mask = _band(array[..., 0], hue_low, hue_high)
    sat_mask = _band(array[..., 1], sat_low, sat_high)
    val_mask = _band(array[..., 2], val_low, val_high)
    return hue_mask & sat_mask & val_mask


def border_mask(
    shape: tuple[int, int],
    thickness: int,
) -> np.ndarray:
    """Return a boolean mask covering a frame of the given thickness.

    Parameters
    ----------
    shape:
        Target shape as ``(height, width)``.
    thickness:
        Frame width in pixels, at least 1.

    Returns
    -------
    numpy.ndarray
        ``bool`` array of shape ``shape``.

    Raises
    ------
    ValueError
        For a non-positive thickness or an empty shape.
    """
    height, width = int(shape[0]), int(shape[1])
    band = int(thickness)
    if band < 1:
        raise ValueError(f"thickness must be >= 1, got {thickness!r}")
    mask = np.zeros((height, width), dtype=bool)
    rows = min(band, height)
    cols = min(band, width)
    mask[:rows, :] = True
    mask[height - rows:, :] = True
    mask[:, :cols] = True
    mask[:, width - cols:] = True
    return mask


@dataclass(frozen=True)
class BackgroundModel:
    """Robust statistical model of the background colours.

    Attributes
    ----------
    modes:
        ``(K, 3)`` array of Lab colour centres, one per mode.
    spreads:
        ``(K,)`` array of robust (MAD based) spreads, at least 1.
    n_samples:
        Number of border samples that survived the trimming.
    hsv_median:
        Median HSV of the kept samples, as a 3-tuple of floats.
    """

    modes: np.ndarray
    spreads: np.ndarray
    n_samples: int
    hsv_median: tuple[float, float, float]


def _trim(
    samples: np.ndarray,
    center: np.ndarray,
    trim_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the kept mask and robust spread of samples around a center."""
    distances = np.linalg.norm(samples - center.reshape(1, 3), axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    robust_sigma = MAD_SCALE * mad
    keep = distances <= median_distance + float(trim_k) * robust_sigma
    if not np.any(keep):
        keep = np.ones_like(distances, dtype=bool)
    return keep, max(robust_sigma, 1.0)


def _farthest_point_init(
    samples: np.ndarray,
    k: int,
) -> np.ndarray:
    """Pick ``k`` deterministic initial centres (farthest-point rule)."""
    centers = [np.median(samples, axis=0)]
    while len(centers) < k:
        distances = np.min(
            np.stack(
                [
                    np.linalg.norm(samples - c.reshape(1, 3), axis=1)
                    for c in centers
                ],
                axis=1,
            ),
            axis=1,
        )
        centers.append(samples[int(np.argmax(distances))].copy())
    return np.stack(centers, axis=0)


def _kmeans(
    samples: np.ndarray,
    k: int,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a deterministic Lloyd k-means with farthest-point initialisation.

    Parameters
    ----------
    samples:
        ``(N, 3)`` array of Lab samples.
    k:
        Number of clusters.
    max_iter:
        Maximum number of assignment steps.

    Returns
    -------
    tuple
        ``(centres, assignment)`` with the assignment given as an
        ``int32`` array of cluster indices.
    """
    centers = _farthest_point_init(samples, k)
    assignment = np.zeros(samples.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        distances = np.stack(
            [
                np.linalg.norm(samples - c.reshape(1, 3), axis=1)
                for c in centers
            ],
            axis=1,
        )
        new_assignment = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for index in range(k):
            members = samples[assignment == index]
            if members.size:
                centers[index] = np.median(members, axis=0)
    return centers, assignment


def estimate_background_model(
    rgb: np.ndarray,
    cfg: PipelineConfig,
) -> BackgroundModel:
    """Fit the background colour model on the border of an image.

    The outer frame of thickness ``max(1, round(border_frac * min(H, W)))``
    is sampled, converted to Lab and reduced to one or two modes.  Samples
    further than ``median_distance + trim_k * 1.4826 * MAD`` from a centre
    are trimmed away and the centre is recomputed on the survivors, so a few
    foreground pixels touching the border cannot drag the model.

    Parameters
    ----------
    rgb:
        ``uint8`` RGB image.
    cfg:
        Validated :class:`~bgremover.config.PipelineConfig`.

    Returns
    -------
    BackgroundModel
        The Lab centres, their robust spreads, the number of kept samples
        and the median HSV of the kept samples.

    Raises
    ------
    ValueError
        If the image or the configuration is invalid.
    """
    array = _as_rgb(rgb)
    cfg.validate()
    height, width = array.shape[:2]
    thickness = max(1, round(cfg.border_frac * min(height, width)))
    frame = border_mask((height, width), thickness)
    lab_all = rgb_to_lab(array)
    samples = lab_all[frame]
    if samples.size == 0:
        raise ValueError("cannot sample the image border")
    hsv_all = rgb_to_hsv(array)[frame]

    centers: list[np.ndarray] = []
    spreads: list[float] = []
    kept_all: list[np.ndarray] = []
    if cfg.n_background_modes == 2 and samples.shape[0] >= 4:
        initial, assignment = _kmeans(samples, 2)
        for index in range(2):
            members = samples[assignment == index]
            if members.shape[0] < 0.05 * samples.shape[0]:
                continue
            keep, spread = _trim(members, np.median(members, axis=0),
                                 cfg.trim_k)
            kept = members[keep]
            centers.append(np.median(kept, axis=0))
            spreads.append(spread)
            kept_all.append(kept)
    if not centers:
        center = np.median(samples, axis=0)
        keep, spread = _trim(samples, center, cfg.trim_k)
        kept = samples[keep]
        centers.append(np.median(kept, axis=0))
        spreads.append(spread)
        kept_all.append(kept)

    kept_samples = np.concatenate(kept_all, axis=0)
    distances = np.linalg.norm(
        samples - np.median(kept_samples, axis=0).reshape(1, 3), axis=1
    )
    kept_mask = distances <= float(np.max(spreads)) * 2.0 + 2.0
    hsv_kept = hsv_all[kept_mask]
    if hsv_kept.size == 0:
        hsv_kept = hsv_all
    LOGGER.debug(
        "background model: %d mode(s) from %d samples",
        len(centers),
        kept_samples.shape[0],
    )
    return BackgroundModel(
        modes=np.stack(centers, axis=0).astype(np.float32),
        spreads=np.array(spreads, dtype=np.float32),
        n_samples=int(kept_samples.shape[0]),
        hsv_median=(
            float(np.median(hsv_kept[:, 0])),
            float(np.median(hsv_kept[:, 1])),
            float(np.median(hsv_kept[:, 2])),
        ),
    )


def background_distance_map(
    rgb_or_lab: np.ndarray,
    model: BackgroundModel,
    *,
    input_is_lab: bool = False,
) -> np.ndarray:
    """Return the normalised distance to the closest background mode.

    ``d = min_k ||lab(p) - mode_k||`` divided by
    :data:`DISTANCE_NORM` (a Lab distance of 100 means "completely
    different") and clipped to ``[0, 1]``.

    Parameters
    ----------
    rgb_or_lab:
        ``uint8`` RGB image or, when ``input_is_lab`` is true, a float Lab
        image.
    model:
        Background model from :func:`estimate_background_model`.
    input_is_lab:
        Set to true when ``rgb_or_lab`` already holds Lab values.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(H, W)`` with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        For a bad image or a model without modes.
    """
    if input_is_lab:
        lab = np.asarray(rgb_or_lab, dtype=np.float32)
        if lab.ndim != 3 or lab.shape[2] != 3:
            raise ValueError(
                f"expected an (H, W, 3) Lab image, got {lab.shape}"
            )
    else:
        lab = rgb_to_lab(rgb_or_lab)
    modes = np.asarray(model.modes, dtype=np.float32)
    if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] == 0:
        raise ValueError("the background model must contain at least one mode")
    distances = np.stack(
        [euclidean_color_distance(lab, mode) for mode in modes], axis=0
    )
    return np.clip(distances.min(axis=0) / DISTANCE_NORM, 0.0,
                    1.0).astype(np.float32)
