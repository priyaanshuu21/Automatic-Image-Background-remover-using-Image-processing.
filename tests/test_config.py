"""Tests for the pipeline configuration object."""

from __future__ import annotations

import dataclasses

import pytest

from bgremover.config import PipelineConfig


def test_defaults_are_valid() -> None:
    """The default configuration must pass validation."""
    config = PipelineConfig()
    assert config.validate() is None
    assert config.work_max_side == 800
    assert config.denoise == "gaussian"
    assert config.grow_connectivity == 8


@pytest.mark.parametrize("connectivity", [0, 1, 2, 3, 5, 6, 7, 9, 16])
def test_invalid_connectivity_raises(connectivity: int) -> None:
    """Connectivity outside ``{4, 8}`` must be rejected."""
    with pytest.raises(ValueError):
        PipelineConfig(grow_connectivity=connectivity)


@pytest.mark.parametrize("ksize", [0, -1, 2, 4, 6, 8])
def test_invalid_ksize_raises(ksize: int) -> None:
    """An even or non-positive median kernel size must be rejected."""
    with pytest.raises(ValueError):
        PipelineConfig(denoise_ksize=ksize)


@pytest.mark.parametrize(
    "field", ["border_frac", "min_object_frac", "fill_hole_frac"],
)
@pytest.mark.parametrize("value", [0.0, -0.1, 1.0, 1.5])
def test_invalid_fractions_raise(field: str, value: float) -> None:
    """Fractions outside ``(0, 1)`` must be rejected."""
    with pytest.raises(ValueError):
        PipelineConfig(**{field: value})


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5])
def test_invalid_hysteresis_ratio_raises(value: float) -> None:
    """``hysteresis_low_ratio`` must lie in ``(0, 1]``."""
    with pytest.raises(ValueError):
        PipelineConfig(hysteresis_low_ratio=value)


@pytest.mark.parametrize("value", ["", "bilateral", "GAUSSIAN", "box"])
def test_invalid_denoise_name_raises(value: str) -> None:
    """An unknown denoiser name must be rejected."""
    with pytest.raises(ValueError):
        PipelineConfig(denoise=value)


@pytest.mark.parametrize("modes", [0, 3, -1])
def test_invalid_mode_count_raises(modes: int) -> None:
    """``n_background_modes`` must be 1 or 2."""
    with pytest.raises(ValueError):
        PipelineConfig(n_background_modes=modes)


def test_canny_thresholds_must_be_ordered() -> None:
    """``canny_low`` must be strictly smaller than ``canny_high``."""
    with pytest.raises(ValueError):
        PipelineConfig(canny_low=0.5, canny_high=0.1)
    with pytest.raises(ValueError):
        PipelineConfig(canny_low=0.3, canny_high=0.3)


def test_non_positive_geometry_rejected() -> None:
    """Radii and scales that break the filters must be rejected."""
    for kwargs in (
        {"work_max_side": 4},
        {"gaussian_sigma": 0.0},
        {"grow_tolerance": -1.0},
        {"trim_k": 0.0},
        {"morph_open_radius": -1},
        {"morph_close_radius": -2},
        {"keep_largest": -1},
        {"feather_sigma": -0.5},
        {"band_width": -1},
    ):
        with pytest.raises(ValueError):
            PipelineConfig(**kwargs)


def test_with_overrides_returns_new_object() -> None:
    """``with_overrides`` must copy and leave the original untouched."""
    original = PipelineConfig()
    updated = original.with_overrides(grow_tolerance=25.0)
    assert updated is not original
    assert updated.grow_tolerance == 25.0
    assert original.grow_tolerance == 14.0
    assert isinstance(updated, PipelineConfig)


def test_with_overrides_rejects_invalid_values() -> None:
    """Overrides are validated exactly like direct construction."""
    with pytest.raises(ValueError):
        PipelineConfig().with_overrides(grow_connectivity=3)


def test_with_overrides_rejects_unknown_fields() -> None:
    """Unknown field names surface as ``TypeError``."""
    with pytest.raises(TypeError):
        PipelineConfig().with_overrides(not_a_field=1)


def test_config_is_frozen() -> None:
    """The dataclass is immutable."""
    config = PipelineConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.grow_tolerance = 1.0  # type: ignore[misc]


def test_field_count_is_stable() -> None:
    """Guards against accidentally dropping a documented field."""
    names = {field.name for field in
             dataclasses.fields(PipelineConfig)}
    expected = {
        "work_max_side", "denoise", "denoise_ksize", "gaussian_sigma",
        "border_frac", "n_background_modes", "trim_k", "grow_tolerance",
        "grow_connectivity", "use_edge_barrier", "canny_low", "canny_high",
        "hysteresis_low_ratio", "morph_open_radius", "morph_close_radius",
        "min_object_frac", "fill_hole_frac", "keep_largest",
        "feather_sigma", "band_width",
    }
    assert names == expected
