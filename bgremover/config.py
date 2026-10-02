"""Configuration object for the background removal pipeline.

The pipeline is fully parameterised by :class:`PipelineConfig`.  The object is
frozen so that a configuration can be shared safely between the pipeline, the
CLI and the GUI without any risk of accidental mutation.  Every field is
validated on construction and again by :meth:`PipelineConfig.validate`, which
makes a mis-configured pipeline impossible to build.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

_DENOISE_CHOICES = ("gaussian", "median", "none")
_CONNECTIVITY_CHOICES = (4, 8)
_MODE_CHOICES = (1, 2)


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable set of parameters driving the pipeline.

    Attributes
    ----------
    work_max_side:
        Longest side, in pixels, of the working resolution.  The original
        image is resized to at most this size for processing and the mask is
        upscaled again at the end.
    denoise:
        Pre-filter applied before segmentation: ``"gaussian"``, ``"median"``
        or ``"none"``.
    denoise_ksize:
        Odd side length of the median filter window (1 disables it).
    gaussian_sigma:
        Standard deviation, in pixels, of the Gaussian pre-filter.
    border_frac:
        Thickness of the sampled image border expressed as a fraction of
        ``min(height, width)``.  Used to fit the background colour model.
    n_background_modes:
        Number of background colour clusters (1 or 2).  Use 2 for images with
        a two-tone background.
    trim_k:
        Number of robust standard deviations used to trim background
        outliers (median + MAD trimming).
    grow_tolerance:
        Maximum Euclidean distance in CIELAB units between a candidate pixel
        and the region reference colour for region growing.
    grow_connectivity:
        Neighbourhood connectivity used by region growing: 4 or 8.
    use_edge_barrier:
        When true, Canny edges stop the background region growing.
    canny_low, canny_high:
        Canny thresholds expressed as fractions of the maximum gradient
        magnitude.
    hysteresis_low_ratio:
        Weak threshold multiplier applied to the Otsu threshold.
    morph_open_radius:
        Radius of the elliptical structuring element used by the opening.
    morph_close_radius:
        Radius of the elliptical structuring element used by the closing.
    min_object_frac:
        Minimum foreground area, as a fraction of the image, for a
        connected component to be kept.
    fill_hole_frac:
        Maximum area, as a fraction of the image, of a hole that is filled.
    keep_largest:
        Number of largest connected components kept in the final mask
        (0 keeps every component).
    feather_sigma:
        Gaussian sigma, in pixels, used to soften the alpha matte.
    band_width:
        Width, in pixels, of the feathered band around the mask boundary.
    """

    work_max_side: int = 800
    denoise: str = "gaussian"
    denoise_ksize: int = 5
    gaussian_sigma: float = 1.2
    border_frac: float = 0.04
    n_background_modes: int = 1
    trim_k: float = 2.5
    grow_tolerance: float = 14.0
    grow_connectivity: int = 8
    use_edge_barrier: bool = True
    canny_low: float = 0.08
    canny_high: float = 0.20
    hysteresis_low_ratio: float = 0.5
    morph_open_radius: int = 2
    morph_close_radius: int = 4
    min_object_frac: float = 0.002
    fill_hole_frac: float = 0.01
    keep_largest: int = 1
    feather_sigma: float = 1.5
    band_width: int = 4

    def __post_init__(self) -> None:
        """Validate the freshly created configuration."""
        self.validate()

    def validate(self) -> None:
        """Check every field and raise ``ValueError`` when out of range.

        Raises
        ------
        ValueError
            If any field is outside its documented domain.
        """
        if int(self.work_max_side) < 16:
            raise ValueError(
                f"work_max_side must be >= 16, got {self.work_max_side!r}"
            )
        if self.denoise not in _DENOISE_CHOICES:
            raise ValueError(
                f"denoise must be one of {_DENOISE_CHOICES}, got "
                f"{self.denoise!r}"
            )
        if int(self.denoise_ksize) < 1 or int(self.denoise_ksize) % 2 == 0:
            raise ValueError(
                "denoise_ksize must be a positive odd integer, got "
                f"{self.denoise_ksize!r}"
            )
        if float(self.gaussian_sigma) <= 0.0:
            raise ValueError(
                f"gaussian_sigma must be > 0, got {self.gaussian_sigma!r}"
            )
        if not 0.0 < float(self.border_frac) < 0.5:
            raise ValueError(
                "border_frac must lie in (0, 0.5), got "
                f"{self.border_frac!r}"
            )
        if int(self.n_background_modes) not in _MODE_CHOICES:
            raise ValueError(
                f"n_background_modes must be 1 or 2, got "
                f"{self.n_background_modes!r}"
            )
        if float(self.trim_k) <= 0.0:
            raise ValueError(f"trim_k must be > 0, got {self.trim_k!r}")
        if float(self.grow_tolerance) <= 0.0:
            raise ValueError(
                f"grow_tolerance must be > 0, got {self.grow_tolerance!r}"
            )
        if int(self.grow_connectivity) not in _CONNECTIVITY_CHOICES:
            raise ValueError(
                f"grow_connectivity must be 4 or 8, got "
                f"{self.grow_connectivity!r}"
            )
        for name in ("canny_low", "canny_high"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{name} must lie in (0, 1], got {value!r}"
                )
        if float(self.canny_low) >= float(self.canny_high):
            raise ValueError(
                f"canny_low ({self.canny_low!r}) must be smaller than "
                f"canny_high ({self.canny_high!r})"
            )
        if not 0.0 < float(self.hysteresis_low_ratio) <= 1.0:
            raise ValueError(
                "hysteresis_low_ratio must lie in (0, 1], got "
                f"{self.hysteresis_low_ratio!r}"
            )
        if int(self.morph_open_radius) < 0:
            raise ValueError(
                f"morph_open_radius must be >= 0, got "
                f"{self.morph_open_radius!r}"
            )
        if int(self.morph_close_radius) < 0:
            raise ValueError(
                f"morph_close_radius must be >= 0, got "
                f"{self.morph_close_radius!r}"
            )
        for name in ("min_object_frac", "fill_hole_frac"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"{name} must lie in (0, 1), got {value!r}"
                )
        if int(self.keep_largest) < 0:
            raise ValueError(
                f"keep_largest must be >= 0, got {self.keep_largest!r}"
            )
        if float(self.feather_sigma) < 0.0:
            raise ValueError(
                f"feather_sigma must be >= 0, got {self.feather_sigma!r}"
            )
        if int(self.band_width) < 0:
            raise ValueError(
                f"band_width must be >= 0, got {self.band_width!r}"
            )

    def with_overrides(self, **kwargs: object) -> "PipelineConfig":
        """Return a validated copy of this configuration.

        Parameters
        ----------
        **kwargs:
            Field names to override.  Unknown names raise ``TypeError``.

        Returns
        -------
        PipelineConfig
            A new, validated configuration.  The receiver is left untouched.

        Raises
        ------
        ValueError
            If an overridden value is out of range.
        """
        return dataclasses.replace(self, **kwargs)  # type: ignore[arg-type]
