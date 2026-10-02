"""End-to-end automatic background removal pipeline.

The orchestrator chains every classical stage — working-resolution resize,
denoising, Lab background modelling, Canny edge barriers, border-seeded
region growing, mask cleanup, alpha feathering, halo decontamination and
full-resolution RGBA export — and records each intermediate result so the
whole decision chain stays inspectable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from bgremover.color import (
    background_distance_map,
    border_mask,
    estimate_background_model,
    rgb_to_lab,
)
from bgremover.config import PipelineConfig
from bgremover.edges import canny, edge_barrier
from bgremover.filters import denoise
from bgremover.fundamentals import resize_max_side, to_gray
from bgremover.io import load_image, save_image
from bgremover.morphology import fill_holes, remove_small_objects
from bgremover.pixels import label_components
from bgremover.refine import (
    alpha_from_mask,
    compose_rgba,
    decontaminate_edges,
    upscale_mask,
)
from bgremover.segmentation import region_growing

LOGGER = logging.getLogger(__name__)

STAGE_KEYS: tuple[str, ...] = (
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
)

# The normalised colour distance halfway to "completely different".
SEED_DISTANCE = 0.5


@dataclass
class PipelineResult:
    """Everything one pipeline run produced.

    Attributes
    ----------
    stages:
        Mapping with exactly the keys of :data:`STAGE_KEYS`: the
        original input, the working-resolution image, its denoised
        version, the normalised background distance, the Canny edges,
        the raw distance seed mask, the grown background region, the
        cleaned foreground mask, the feathered alpha matte (working
        resolution) and the final full-resolution RGBA image.
    scale:
        The resize factor applied to reach the working resolution.
    config:
        The configuration the run used.
    """

    stages: dict[str, np.ndarray] = field(default_factory=dict)
    scale: float = 1.0
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        """Check that every required stage is present."""
        missing = [key for key in STAGE_KEYS if key not in self.stages]
        if missing:
            raise ValueError(f"missing pipeline stages: {missing}")

    @property
    def rgba(self) -> np.ndarray:
        """The final full-resolution ``uint8`` RGBA image."""
        return self.stages["rgba"]

    @property
    def alpha(self) -> np.ndarray:
        """The working-resolution ``float32`` alpha matte."""
        return self.stages["alpha"]

    @property
    def mask(self) -> np.ndarray:
        """The cleaned boolean foreground mask (working resolution)."""
        return self.stages["cleaned_mask"]


class BackgroundRemovalPipeline:
    """Configurable classical background removal pipeline.

    Parameters
    ----------
    cfg:
        Pipeline configuration; a default one is built when omitted.

    Raises
    ------
    ValueError
        If the configuration is invalid.
    """

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        """Store a validated copy of the configuration."""
        self.cfg = cfg if cfg is not None else PipelineConfig()
        self.cfg.validate()

    def run(self, rgb: np.ndarray) -> PipelineResult:
        """Remove the background of an RGB image.

        The background is assumed to touch the image border: it is
        modelled in CIELAB, grown from border seeds (stopped by Canny
        edges when configured) and everything else is kept as
        foreground.

        Parameters
        ----------
        rgb:
            ``uint8`` RGB array of shape ``(H, W, 3)``.

        Returns
        -------
        PipelineResult
            All ten intermediate stages plus the final RGBA image.

        Raises
        ------
        ValueError
            If the input is not an ``uint8`` RGB image.
        """
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"expected an RGB image of shape (H, W, 3), "
                f"got {array.shape}"
            )
        if array.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got {array.dtype}")
        if array.size == 0:
            raise ValueError("image must not be empty")
        cfg = self.cfg
        full_shape = array.shape[:2]

        resized, scale = resize_max_side(array, cfg.work_max_side)
        denoised = denoise(resized, cfg)
        model = estimate_background_model(denoised, cfg)
        distance = background_distance_map(denoised, model)

        gray = to_gray(denoised)
        edges = canny(gray, cfg.canny_low, cfg.canny_high)
        if cfg.use_edge_barrier:
            barrier = edge_barrier(edges, radius=1)
        else:
            barrier = np.zeros(gray.shape, dtype=bool)

        seed_mask = distance >= SEED_DISTANCE

        height, width = gray.shape
        thickness = max(1, round(cfg.border_frac * min(height, width)))
        frame = border_mask((height, width), thickness)
        seeds = [tuple(position) for position in np.argwhere(frame)]
        grown = region_growing(
            seeds,
            rgb_to_lab(denoised),
            tolerance=cfg.grow_tolerance,
            connectivity=cfg.grow_connectivity,
            allowed=~barrier,
            reference="seed",
        )

        foreground = ~grown
        # Barrier pixels are ambiguous by construction: region growing
        # was forbidden from entering them.  Resolve them with the
        # colour model instead: a barrier pixel whose Lab distance is
        # clearly background rejoins the background.
        foreground = foreground & ~(barrier & (distance < SEED_DISTANCE))
        if not foreground.any():
            LOGGER.debug("growing covered everything; using seed mask")
            foreground = seed_mask.copy()
        area = float(height * width)
        smallest = max(1, round(cfg.min_object_frac * area))
        cleaned = remove_small_objects(foreground, min_size=smallest)
        cleaned = fill_holes(cleaned)
        if int(cfg.keep_largest) > 0 and cleaned.any():
            labels, _ = label_components(cleaned, connectivity=8)
            sizes = np.bincount(labels.ravel())
            order = np.argsort(sizes[1:])[::-1] + 1
            keep = set(int(label) for label in
                       order[: int(cfg.keep_largest)].tolist())
            cleaned = np.isin(labels, sorted(keep))

        alpha = alpha_from_mask(
            cleaned, cfg.feather_sigma, cfg.band_width
        )
        full_alpha = upscale_mask(alpha, full_shape)
        colours = decontaminate_edges(array, full_alpha)
        rgba = compose_rgba(colours, full_alpha)

        stages = {
            "input": array.copy(),
            "resized": resized,
            "denoised": denoised,
            "distance": distance,
            "edges": edges,
            "seed_mask": seed_mask,
            "grown_mask": grown,
            "cleaned_mask": cleaned,
            "alpha": alpha,
            "rgba": rgba,
        }
        LOGGER.debug(
            "pipeline done: scale=%.3f foreground=%.1f%%",
            scale,
            100.0 * float(cleaned.mean()),
        )
        return PipelineResult(stages=stages, scale=float(scale),
                              config=cfg)

    def run_file(self, source: str, destination: str) -> PipelineResult:
        """Remove the background of an image file and save the RGBA.

        Parameters
        ----------
        source:
            Path of a readable image file.
        destination:
            Path of the ``.png``/``.tiff`` file receiving the RGBA
            result (parent folders are created as needed).

        Returns
        -------
        PipelineResult
            All ten intermediate stages plus the final RGBA image.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        ValueError
            If either path is unusable.
        """
        result = self.run(load_image(source))
        save_image(destination, result.rgba)
        return result
