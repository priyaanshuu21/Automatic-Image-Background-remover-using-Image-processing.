"""Tests for the end-to-end background removal pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from bgremover.config import PipelineConfig
from bgremover.io import load_image
from bgremover.pipeline import (
    STAGE_KEYS,
    BackgroundRemovalPipeline,
    PipelineResult,
)


def test_stage_keys_are_exactly_ten() -> None:
    """The result contract holds ten named stages."""
    assert len(STAGE_KEYS) == 10
    assert set(STAGE_KEYS) == {
        "input",
        "resized",
        "denoised",
        "distance",
        "edges",
        "seed_mask",
        "grown_mask",
        "cleaned_mask",
        "alpha",
        "rgba",
    }


def test_result_rejects_missing_stages() -> None:
    """A result without all ten stages cannot be built."""
    with pytest.raises(ValueError):
        PipelineResult(stages={"rgba": np.zeros((2, 2, 4), np.uint8)})


def test_pipeline_stage_shapes_and_dtypes(circle_image) -> None:
    """Every stage has the documented shape and dtype."""
    result = BackgroundRemovalPipeline().run(circle_image)
    stages = result.stages
    assert set(stages) == set(STAGE_KEYS)
    height, width = circle_image.shape[:2]
    assert stages["input"].shape == (height, width, 3)
    assert stages["resized"].shape == (height, width, 3)
    assert stages["denoised"].shape == (height, width, 3)
    assert stages["denoised"].dtype == np.uint8
    assert stages["distance"].shape == (height, width)
    assert stages["distance"].dtype == np.float32
    assert 0.0 <= float(stages["distance"].min())
    assert float(stages["distance"].max()) <= 1.0
    for key in ("edges", "seed_mask", "grown_mask", "cleaned_mask"):
        assert stages[key].shape == (height, width)
        assert stages[key].dtype == np.bool_
    assert stages["alpha"].shape == (height, width)
    assert stages["alpha"].dtype == np.float32
    assert stages["rgba"].shape == (height, width, 4)
    assert stages["rgba"].dtype == np.uint8
    assert result.scale == 1.0
    assert result.rgba is stages["rgba"]
    assert result.alpha is stages["alpha"]
    assert result.mask is stages["cleaned_mask"]


def test_pipeline_iou_on_fixtures(circle_on_gradient, rect_on_flat) -> None:
    """Both fixtures segment almost perfectly."""
    pipeline = BackgroundRemovalPipeline()
    for image, truth in (circle_on_gradient, rect_on_flat):
        mask = pipeline.run(image).mask
        union = np.count_nonzero(mask | truth)
        score = np.count_nonzero(mask & truth) / union
        assert score >= 0.95


def test_pipeline_is_deterministic(circle_image) -> None:
    """Two runs on the same image agree pixel by pixel."""
    pipeline = BackgroundRemovalPipeline()
    first = pipeline.run(circle_image)
    second = pipeline.run(circle_image)
    assert np.array_equal(first.rgba, second.rgba)
    assert np.array_equal(first.mask, second.mask)


def test_pipeline_configuration_variants(circle_image) -> None:
    """Disabled barrier, kept components and small work size all run."""
    plain = BackgroundRemovalPipeline(
        PipelineConfig(use_edge_barrier=False)
    ).run(circle_image)
    assert plain.rgba.shape == circle_image.shape[:2] + (4,)
    kept = BackgroundRemovalPipeline(
        PipelineConfig(keep_largest=0)
    ).run(circle_image)
    assert kept.mask.any()
    small = BackgroundRemovalPipeline(
        PipelineConfig(work_max_side=100)
    ).run(circle_image)
    assert small.scale == pytest.approx(100 / 160)
    assert small.rgba.shape == circle_image.shape[:2] + (4,)
    assert small.stages["resized"].shape[:2] != circle_image.shape[:2]


def test_pipeline_run_file_roundtrip(circle_image, tmp_path) -> None:
    """Files load as RGB and save back as RGBA with sane alpha."""
    from bgremover.io import save_image

    source = tmp_path / "in.png"
    destination = tmp_path / "nested" / "out.png"
    save_image(source, circle_image)
    result = BackgroundRemovalPipeline().run_file(
        str(source), str(destination)
    )
    assert destination.is_file()
    back = load_image(destination)
    assert back.shape == circle_image.shape
    assert result.rgba[60, 80, 3] == 255
    assert result.rgba[0, 0, 3] == 0
    with pytest.raises(FileNotFoundError):
        BackgroundRemovalPipeline().run_file(
            str(tmp_path / "missing.png"), str(destination)
        )


def test_pipeline_rejects_bad_input() -> None:
    """Non-RGB input is rejected before any work happens."""
    pipeline = BackgroundRemovalPipeline()
    with pytest.raises(ValueError):
        pipeline.run(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        pipeline.run(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline.run(np.zeros((0, 0, 3), dtype=np.uint8))
