"""Tests for colour spaces, colour slicing and the background colour model."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.color import (
    BackgroundModel,
    background_distance_map,
    border_mask,
    estimate_background_model,
    euclidean_color_distance,
    hsv_channel_images,
    hsv_slice,
    hsv_to_opencv_scale,
    hsv_to_rgb,
    lab_to_rgb,
    rgb_to_hsi,
    rgb_to_hsv,
    rgb_to_lab,
)
from bgremover.config import PipelineConfig

PRIMARIES = np.array(
    [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255],
      [128, 128, 128], [10, 20, 30]]],
    dtype=np.uint8,
)


def test_rgb_to_hsv_matches_opencv(random_rgb) -> None:
    """Hue, saturation and value agree with ``RGB2HSV_FULL``."""
    strip = random_rgb.reshape(1, -1, 3)
    mine = rgb_to_hsv(strip).reshape(-1, 3)
    reference = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV_FULL).reshape(-1, 3)
    scaled_hue = np.mod(mine[:, 0] / 360.0 * 256.0, 256.0)
    hue_delta = np.abs((scaled_hue - reference[:, 0] + 128.0) % 256.0
                       - 128.0)
    assert float(hue_delta.max()) <= 1.0
    assert float(np.abs(mine[:, 1] * 255.0 - reference[:, 1]).max()) <= 1.0
    assert float(np.abs(mine[:, 2] * 255.0 - reference[:, 2]).max()) <= 1.0


def test_rgb_to_hsv_primaries_and_gray() -> None:
    """Primary hues are 0, 120 and 240 degrees; gray has no saturation."""
    hsv = rgb_to_hsv(PRIMARIES)[0]
    assert hsv[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert hsv[1, 0] == pytest.approx(120.0, abs=1e-3)
    assert hsv[2, 0] == pytest.approx(240.0, abs=1e-3)
    assert hsv[0, 1] == pytest.approx(1.0)
    assert hsv[3, 1] == pytest.approx(0.0, abs=1e-6)
    assert hsv[3, 2] == pytest.approx(1.0)
    assert hsv[4, 1] == pytest.approx(0.0, abs=1e-6)
    assert hsv[4, 0] == pytest.approx(0.0, abs=1e-6)
    assert hsv[4, 2] == pytest.approx(128.0 / 255.0, abs=1e-6)


def test_hsv_round_trip(random_rgb) -> None:
    """HSV -> RGB inverts RGB -> HSV within one gray level."""
    back = hsv_to_rgb(rgb_to_hsv(random_rgb))
    assert back.dtype == np.uint8
    assert int(np.abs(back.astype(int) - random_rgb.astype(int)).max()) <= 1


def test_hsv_to_opencv_scale_and_channels(rect_image) -> None:
    """The OpenCV layout halves the hue and rescales S and V."""
    hsv = rgb_to_hsv(rect_image)
    scaled = hsv_to_opencv_scale(hsv)
    channels = hsv_channel_images(rect_image)
    assert scaled.shape == rect_image.shape
    assert int(scaled[..., 0].max()) <= 180
    assert set(channels) == {"H", "S", "V"}
    for image in channels.values():
        assert image.shape == rect_image.shape[:2]
        assert image.dtype == np.uint8
    assert np.array_equal(channels["V"], rect_image[..., 1])


def test_hsi_primaries_and_gray() -> None:
    """HSI hues match the primaries; white has unit intensity."""
    hsi = rgb_to_hsi(PRIMARIES)[0]
    assert hsi[0, 0] == pytest.approx(0.0, abs=1e-2)
    assert hsi[1, 0] == pytest.approx(120.0, abs=1e-2)
    assert hsi[2, 0] == pytest.approx(240.0, abs=1e-2)
    assert hsi[3, 2] == pytest.approx(1.0)
    assert hsi[3, 1] == pytest.approx(0.0, abs=1e-6)
    assert hsi[4, 1] == pytest.approx(0.0, abs=1e-6)
    assert hsi[4, 2] == pytest.approx(128.0 / 255.0, abs=1e-6)


def test_hsi_is_finite(random_rgb) -> None:
    """No NaN or infinity may appear, not even for degenerate colours."""
    hsi = rgb_to_hsi(random_rgb)
    assert np.all(np.isfinite(hsi))
    assert hsi[..., 0].min() >= 0.0
    assert hsi[..., 0].max() <= 360.0
    assert hsi[..., 1].min() >= 0.0
    assert hsi[..., 1].max() <= 1.0


def test_rgb_to_lab_matches_opencv(random_rgb) -> None:
    """CIELAB agrees with OpenCV within one unit per channel."""
    scaled = (random_rgb.astype(np.float32) / 255.0).reshape(1, -1, 3)
    mine = rgb_to_lab(scaled).reshape(-1, 3)
    reference = cv2.cvtColor(scaled, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    assert np.max(np.abs(mine - reference)) <= 1.0


def test_lab_endpoints() -> None:
    """White has ``L = 100`` and black ``L = 0``."""
    lab = rgb_to_lab(PRIMARIES)[0]
    assert lab[3, 0] == pytest.approx(100.0, abs=0.5)
    assert abs(lab[3, 1]) < 1.0 and abs(lab[3, 2]) < 1.0
    assert lab[4, 0] == pytest.approx(53.6, abs=1.0)
    assert abs(lab[4, 1]) < 1.0 and abs(lab[4, 2]) < 1.0
    # a dark colour is far down the lightness axis
    assert lab[5, 0] == pytest.approx(5.9, abs=1.0)
    black = rgb_to_lab(np.zeros((1, 1, 3), dtype=np.uint8))[0, 0]
    assert black[0] == pytest.approx(0.0)


def test_lab_round_trip(random_rgb) -> None:
    """Lab -> RGB inverts RGB -> Lab within a few levels."""
    back = lab_to_rgb(rgb_to_lab(random_rgb))
    assert back.shape == random_rgb.shape
    assert int(np.abs(back.astype(int) - random_rgb.astype(int)).max()) <= 3


def test_color_distance_basics() -> None:
    """Distances are zero for identical colours and follow the 3-4-5 rule."""
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[..., 0] = 3.0
    image[..., 1] = 4.0
    assert float(euclidean_color_distance(image, (3.0, 4.0, 0.0)).max()) \
        == pytest.approx(0.0)
    first = np.zeros((1, 1, 3), dtype=np.float32)
    second = np.full((1, 1, 3), 3.0, dtype=np.float32)
    forward = float(euclidean_color_distance(second, (0.0, 0.0, 0.0))[0, 0])
    backward = float(euclidean_color_distance(first, (3.0, 3.0, 3.0))[0, 0])
    assert forward == pytest.approx(backward)
    assert forward == pytest.approx(np.sqrt(27.0))


def test_color_distance_with_reference_image() -> None:
    """An image reference gives a per-pixel difference map."""
    a = np.zeros((2, 2, 3), dtype=np.float32)
    b = np.full((2, 2, 3), 1.0, dtype=np.float32)
    out = euclidean_color_distance(a, b)
    assert out.shape == (2, 2)
    assert np.allclose(out, np.sqrt(3.0))
    with pytest.raises(ValueError):
        euclidean_color_distance(a, np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        euclidean_color_distance(a, (1.0, 2.0))


def test_hsv_slice_wrap_around() -> None:
    """A hue range such as ``(350, 10)`` wraps around 0 degrees."""
    image = np.zeros((1, 6, 3), dtype=np.uint8)
    hues = [0.0, 5.0, 100.0, 200.0, 350.0, 355.0]
    for index, hue in enumerate(hues):
        image[0, index] = hsv_to_rgb(
            np.array([[[hue, 1.0, 1.0]]], dtype=np.float32)
        )[0, 0]
    hsv = rgb_to_hsv(image)
    mask = hsv_slice(hsv, (350.0, 10.0), (0.5, 1.0), (0.5, 1.0))
    assert list(mask[0]) == [True, True, False, False, True, True]


def test_hsv_slice_isolates_the_red_circle(circle_image) -> None:
    """A red hue slice selects the red disc of the fixture exactly."""
    hsv = rgb_to_hsv(circle_image)
    mask = hsv_slice(hsv, (330.0, 30.0), (0.5, 1.0), (0.3, 1.0))
    assert mask.dtype == np.bool_
    assert int(mask.sum()) > 1000


def test_hsv_slice_rejects_bad_ranges(circle_image) -> None:
    """Reversed saturation or value ranges are rejected."""
    hsv = rgb_to_hsv(circle_image)
    with pytest.raises(ValueError):
        hsv_slice(hsv, (0.0, 30.0), (0.8, 0.2), (0.0, 1.0))
    with pytest.raises(ValueError):
        hsv_slice(hsv, (0.0, 30.0), (0.0, 1.0), (1.0, 0.1))
    with pytest.raises(ValueError):
        hsv_slice(hsv, (-5.0, 30.0), (0.0, 1.0), (0.0, 1.0))
    with pytest.raises(ValueError):
        hsv_slice(np.zeros((4, 4, 3), dtype=np.uint8),
                  (0.0, 30.0), (0.0, 1.0), (0.0, 1.0))


def test_border_mask() -> None:
    """The frame has the requested thickness and touches all four sides."""
    mask = border_mask((10, 12), 2)
    assert mask.shape == (10, 12)
    assert int(mask[0].sum()) == 12
    assert int(mask[:, 0].sum()) == 10
    assert int(mask[5, 5]) == 0
    thick = border_mask((6, 6), 10)
    assert bool(thick.all())
    with pytest.raises(ValueError):
        border_mask((6, 6), 0)


def test_background_model_on_the_gradient(circle_on_gradient) -> None:
    """The model reproduces the true border colour and separates the disc."""
    image, mask = circle_on_gradient
    cfg = PipelineConfig()
    model = estimate_background_model(image, cfg)
    assert isinstance(model, BackgroundModel)
    assert model.modes.shape == (1, 3)
    assert model.spreads.shape == (1,)
    assert model.n_samples > 0
    assert len(model.hsv_median) == 3

    lab = rgb_to_lab(image)
    frame = border_mask(
        image.shape[:2], max(1, round(cfg.border_frac * min(image.shape[:2])))
    )
    truth = np.median(lab[frame], axis=0)
    assert float(np.linalg.norm(model.modes[0] - truth)) < 6.0

    distance = background_distance_map(image, model)
    assert distance.shape == image.shape[:2]
    assert distance.dtype == np.float32
    assert 0.0 <= float(distance.min()) and float(distance.max()) <= 1.0
    assert float(distance[mask].mean()) > 0.3
    assert float(distance[frame].mean()) < 0.1


def test_background_distance_map_accepts_lab(circle_image) -> None:
    """Pre-computed Lab input gives the same map as RGB input."""
    model = estimate_background_model(circle_image, PipelineConfig())
    from_rgb = background_distance_map(circle_image, model)
    from_lab = background_distance_map(rgb_to_lab(circle_image), model,
                                       input_is_lab=True)
    assert np.allclose(from_rgb, from_lab, atol=1e-4)


def test_background_distance_map_rejects_bad_model(circle_image) -> None:
    """A model without modes is rejected."""
    empty = BackgroundModel(
        modes=np.zeros((0, 3), dtype=np.float32),
        spreads=np.zeros(0, dtype=np.float32),
        n_samples=0,
        hsv_median=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        background_distance_map(circle_image, empty)
    with pytest.raises(ValueError):
        background_distance_map(np.zeros((4, 4, 3), np.uint8), empty)


def test_two_mode_background_model() -> None:
    """A two-tone background yields two modes, deterministically."""
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[:, :30] = (40, 40, 200)
    image[:, 30:] = (40, 200, 60)
    cfg = PipelineConfig(n_background_modes=2)
    model = estimate_background_model(image, cfg)
    assert model.modes.shape == (2, 3)
    again = estimate_background_model(image, cfg)
    assert np.array_equal(model.modes, again.modes)
    assert np.array_equal(model.spreads, again.spreads)
    single = estimate_background_model(image, PipelineConfig())
    assert single.modes.shape == (1, 3)


def test_two_modes_fall_back_when_one_cluster_is_tiny() -> None:
    """A sliver of a second colour does not create a second mode."""
    image = np.full((60, 60, 3), 200, dtype=np.uint8)
    image[:2, :2] = (10, 10, 10)
    model = estimate_background_model(
        image, PipelineConfig(n_background_modes=2)
    )
    assert model.modes.shape == (1, 3)


def test_background_model_is_robust_to_outliers(circle_on_gradient) -> None:
    """Five percent of foreign border pixels barely move the model."""
    image, _ = circle_on_gradient
    cfg = PipelineConfig()
    clean = estimate_background_model(image, cfg)
    rng = np.random.default_rng(1)
    polluted = image.copy()
    rows = rng.integers(0, image.shape[0], 200)
    cols = rng.integers(0, image.shape[1], 200)
    polluted[rows, cols] = (255, 0, 0)
    noisy = estimate_background_model(polluted, cfg)
    shift = float(np.linalg.norm(noisy.modes[0] - clean.modes[0]))
    assert shift < 3.0


def test_color_functions_reject_bad_input() -> None:
    """Every colour conversion validates its input."""
    bad = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        rgb_to_hsv(bad)
    with pytest.raises(ValueError):
        rgb_to_hsi(bad)
    with pytest.raises(ValueError):
        rgb_to_lab(bad)
    with pytest.raises(ValueError):
        hsv_to_rgb(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        lab_to_rgb(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        estimate_background_model(np.zeros((4, 4, 3), np.uint8),
                                  PipelineConfig(n_background_modes=3))
