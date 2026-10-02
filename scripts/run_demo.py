"""Run the background removal pipeline on the synthetic samples.

The demo loads the deterministic fixtures, removes the background of each
one, writes the RGBA results plus a stage grid to ``samples/generated/``
(which is ignored by git) and prints the mask IoU against ground truth::

    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "samples" / "generated"
sys.path.insert(0, str(ROOT))

from bgremover.config import PipelineConfig  # noqa: E402
from bgremover.io import save_image  # noqa: E402
from bgremover.pipeline import BackgroundRemovalPipeline  # noqa: E402
from bgremover.viz import save_stage_grid  # noqa: E402


def _load_conftest():
    """Import ``tests/conftest.py`` as a module without pytest magic."""
    path = ROOT / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("bgr_conftest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixtures from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Run the demo on every synthetic sample and report IoU scores."""
    fixtures = _load_conftest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = BackgroundRemovalPipeline(PipelineConfig())
    samples = {
        "circle": fixtures.make_circle_on_gradient(),
        "rect": fixtures.make_rect_on_flat(),
    }
    for name, (image, truth) in samples.items():
        started = time.perf_counter()
        result = pipeline.run(image)
        elapsed = time.perf_counter() - started
        mask = result.mask
        union = np.count_nonzero(mask | truth)
        score = 1.0 if union == 0 else (
            np.count_nonzero(mask & truth) / union
        )
        rgba_path = OUTPUT_DIR / f"demo_rgba_{name}.png"
        grid_path = OUTPUT_DIR / f"demo_stages_{name}.png"
        save_image(rgba_path, result.rgba)
        save_stage_grid(grid_path, result.stages)
        print(
            f"{name}: IoU={score:.4f} "
            f"foreground={100.0 * float(mask.mean()):.1f}% "
            f"{elapsed:.2f}s -> {rgba_path.name}, {grid_path.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
