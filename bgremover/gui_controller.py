"""Headless application logic behind the Tkinter front-end.

:class:`GuiController` owns the image, the pipeline run and every
derived preview, so the Tk view in :mod:`bgremover.gui` stays thin and
all behaviour is testable without ever creating a window.
"""

from __future__ import annotations

import numpy as np

from bgremover.config import PipelineConfig
from bgremover.io import load_image, save_image
from bgremover.pipeline import BackgroundRemovalPipeline, PipelineResult
from bgremover.viz import composite_over_checkerboard, stage_to_rgb

__all__ = ["GuiController"]


class GuiController:
    """Own the image and pipeline state of the desktop app.

    Parameters
    ----------
    cfg:
        Pipeline configuration; a default one is built when omitted.
    """

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        """Create a controller with an empty session."""
        config = cfg if cfg is not None else PipelineConfig()
        config.validate()
        self.cfg = config
        self.pipeline = BackgroundRemovalPipeline(config)
        self.image: np.ndarray | None = None
        self.result: PipelineResult | None = None
        self.source: str = ""

    def load(self, path: str) -> np.ndarray:
        """Load an image file and clear any previous result.

        Parameters
        ----------
        path:
            Path of a readable image file.

        Returns
        -------
        numpy.ndarray
            The loaded ``uint8`` RGB image.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be decoded as an image.
        """
        self.image = load_image(path)
        self.result = None
        self.source = path
        return self.image

    def run(self) -> PipelineResult:
        """Remove the background of the loaded image.

        Returns
        -------
        PipelineResult
            All ten pipeline stages plus the final RGBA image.

        Raises
        ------
        RuntimeError
            If no image has been loaded yet.
        """
        if self.image is None:
            raise RuntimeError("load an image before running")
        self.result = self.pipeline.run(self.image)
        return self.result

    def has_image(self) -> bool:
        """Whether an image is currently loaded."""
        return self.image is not None

    def has_result(self) -> bool:
        """Whether a pipeline result is currently available."""
        return self.result is not None

    def reset(self) -> None:
        """Forget the image and the result of this session."""
        self.image = None
        self.result = None
        self.source = ""

    def _require_result(self) -> PipelineResult:
        """Return the result or raise a helpful error."""
        if self.result is None:
            raise RuntimeError("run the pipeline before previewing")
        return self.result

    def preview_rgba(self) -> np.ndarray:
        """Return the RGBA result composited over a checkerboard."""
        return composite_over_checkerboard(self._require_result().rgba)

    def preview_stage(self, name: str) -> np.ndarray:
        """Return a stage rendered as an ``uint8`` RGB preview.

        Parameters
        ----------
        name:
            One of the ten pipeline stage keys.

        Returns
        -------
        numpy.ndarray
            ``uint8`` RGB array of the stage's own shape.

        Raises
        ------
        RuntimeError
            If no result is available.
        KeyError
            If the stage name is unknown.
        """
        stages = self._require_result().stages
        if name not in stages:
            raise KeyError(f"unknown stage: {name!r}")
        return stage_to_rgb(stages[name])

    def save_rgba(self, path: str) -> None:
        """Save the RGBA result to disk.

        Parameters
        ----------
        path:
            Destination ``.png``/``.tiff`` file.

        Raises
        ------
        RuntimeError
            If no result is available.
        ValueError
            If the destination is unusable.
        """
        save_image(path, self._require_result().rgba)

    def summary(self) -> str:
        """Return a one-line summary of the current session."""
        if self.image is None:
            return "no image loaded"
        text = f"{self.source or 'image'} {self.image.shape[1]}x"
        text += f"{self.image.shape[0]}"
        if self.result is None:
            return text + " (not processed)"
        foreground = 100.0 * float(self.result.mask.mean())
        return text + f" foreground {foreground:.1f}%"
