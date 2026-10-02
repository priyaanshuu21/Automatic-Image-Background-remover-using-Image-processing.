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
    DISTANCE_NORM,
    background_distance_map,
    estimate_zoned_background_model,
    perimeter_zones,
    rgb_to_lab,
    saliency_foreground_seeds,
    select_clean_zones,
)
from bgremover.config import PipelineConfig
from bgremover.edges import canny, combined_edge_response, edge_barrier
from bgremover.filters import denoise
from bgremover.fundamentals import resize_max_side, to_gray
from bgremover.io import load_image, save_image
from bgremover.morphology import (
    dilate,
    erode,
    fill_holes,
    remove_small_objects,
    solidify_foreground,
    structuring_element,
)
from bgremover.pixels import label_components
from bgremover.refine import (
    compose_rgba,
    decontaminate_edges,
    guided_alpha_feathering,
    upscale_mask,
)
from bgremover.segmentation import grow_background, otsu_mask

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
# Absorption ceiling: smooth background variations (soft shadows,
# gradients, blur fringes) live below it while even mid-contrast
# subjects (white mug: ~0.36) stay clearly above.
ABSORB_DISTANCE = 0.25


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
        gray = to_gray(denoised)
        edges = canny(gray, cfg.canny_low, cfg.canny_high)
        model = estimate_zoned_background_model(denoised, cfg, edges)
        distance = background_distance_map(denoised, model)

        if cfg.use_edge_barrier:
            barrier = edge_barrier(edges, radius=2)
        else:
            barrier = np.zeros(gray.shape, dtype=bool)

        # Raw colour foreground: Otsu on the NON-denoised distance map,
        # so blur never shifts the decision boundary.  Topology comes
        # from the smooth chain below; localisation comes from here.
        raw_lab = rgb_to_lab(resized)
        raw_distances = np.stack(
            [
                np.linalg.norm(raw_lab - mode.reshape(1, 1, 3), axis=2)
                for mode in model.modes.astype(np.float32)
            ],
            axis=0,
        )
        raw_distance = np.clip(
            raw_distances.min(axis=0) / DISTANCE_NORM, 0.0, 1.0
        )
        raw_levels = np.rint(raw_distance * 255.0).astype(np.uint8)
        seed_mask, _ = otsu_mask(raw_levels)

        height, width = gray.shape
        lab = rgb_to_lab(denoised)
        thickness = max(1, round(cfg.border_frac * min(height, width)))
        zones = perimeter_zones((height, width), thickness)
        clean = select_clean_zones(lab, zones, edges)
        # Verify the surviving zones against the fitted model: a
        # uniform subject region touching the border (clean by
        # variance, edgeless by luck) still disagrees with every
        # background mode and must not supply seeds.
        spread = float(model.spreads.max())
        seed_bound = spread * 2.0 + 2.0
        seed_zones = []
        for zid in clean:
            zone_mean = lab[zones == zid].mean(axis=0)
            nearest = float(np.min(np.linalg.norm(
                model.modes - zone_mean.reshape(1, 3), axis=1
            )))
            if nearest <= seed_bound:
                seed_zones.append(zid)
        if not seed_zones:
            seed_zones = clean
        seeds = [
            tuple(position)
            for position in np.argwhere(np.isin(zones, seed_zones))
        ]
        energy = combined_edge_response(
            gray, cfg.canny_low, cfg.canny_high
        )
        adaptive_tolerance = max(cfg.grow_tolerance, cfg.trim_k * spread)
        grown = grow_background(
            lab,
            seeds,
            energy,
            tolerance=adaptive_tolerance,
            max_edge=0.12,
            connectivity=cfg.grow_connectivity,
            allowed=~barrier,
            reference="seed",
        )

        foreground = ~grown
        # Smooth near-background areas (soft shadows, gradients, blur
        # fringes) rejoin the background when they connect to grown
        # background outside the barrier.  The absorption ceiling is
        # deliberately looser than the growth gate; the chromatic gate
        # does the discriminating instead, because shadows preserve
        # the background hue while real subjects shift it.
        chroma = np.min(
            np.stack(
                [
                    np.linalg.norm(
                        lab[..., 1:] - mode.reshape(1, 1, 2), axis=2
                    )
                    for mode in model.modes[:, 1:]
                ],
                axis=0,
            ),
            axis=0,
        )
        chroma_tolerance = max(5.0, spread)
        low_region = (
            (distance < ABSORB_DISTANCE)
            & ~barrier
            & (chroma <= chroma_tolerance)
        )
        joint, _ = label_components(grown | low_region, connectivity=8)
        grown_labels = set(np.unique(joint[grown]).tolist()) - {0}
        if grown_labels:
            grown = np.isin(joint, sorted(grown_labels))
            foreground = ~grown
        # Barrier pixels are ambiguous by construction: region growing
        # was forbidden from entering them.  Resolve them with the
        # colour model instead: a barrier pixel whose Lab distance is
        # clearly background rejoins the background, unless the Otsu
        # seed already claims it (mid-contrast subject structures such
        # as a white handle live inside the barrier at dist < 0.5).
        foreground = foreground & ~(
            barrier & (distance < SEED_DISTANCE) & ~seed_mask
        )
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
        # Definite foreground from the saliency prior, then topological
        # solidification so enclosed regions can never stay hollow.
        # Saliency never overrides the barrier: edge pixels were already
        # resolved by the colour fallback above.  The nearest mode
        # counts, so neither half of a two-tone background is salient,
        # and the floor sits above lighting/shadow variations.
        contrast = max(20.0, 2.0 * spread)
        salient = saliency_foreground_seeds(
            lab, model.modes, contrast
        ) & ~barrier
        solid_radius = min(6, max(4, int(cfg.morph_close_radius)))
        cleaned = solidify_foreground(
            cleaned | salient, close_radius=solid_radius
        )
        cleaned = remove_small_objects(cleaned, min_size=smallest)
        # Localise the boundary with raw colours: every pixel in a
        # 2-pixel band around the smoothed topology takes the raw Otsu
        # seed label, undoing denoise-blur spillover on both sides.
        edge_element = structuring_element("disk", 2)
        band = dilate(cleaned, kernel=edge_element).astype(bool)
        band &= ~erode(cleaned, kernel=edge_element).astype(bool)
        if band.any():
            cleaned = np.where(band, seed_mask, cleaned)
        cleaned = remove_small_objects(cleaned, min_size=smallest)

        feather_radius = min(5, max(3, int(cfg.band_width)))
        alpha = guided_alpha_feathering(cleaned, radius=feather_radius)
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
