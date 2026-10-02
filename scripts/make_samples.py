"""Write the synthetic test fixtures to ``samples/generated/``.

The generated folder is ignored by git; run this script whenever real PNG
files are needed for the CLI, the GUI or the report figures::

    .\\.venv\\Scripts\\python.exe scripts\\make_samples.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "samples" / "generated"
sys.path.insert(0, str(ROOT))

from bgremover.io import save_image  # noqa: E402


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
    """Write every synthetic sample and print one line per file."""
    fixtures = _load_conftest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    circle_image, circle_mask = fixtures.make_circle_on_gradient()
    rect_image, rect_mask = fixtures.make_rect_on_flat()
    noisy = fixtures.noisy_variant(circle_image, 8.0)

    save_image(OUTPUT_DIR / "circle_on_gradient.png", circle_image)
    save_image(OUTPUT_DIR / "circle_mask.png", circle_mask.astype(np.uint8)
               * 255)
    save_image(OUTPUT_DIR / "rect_on_flat.png", rect_image)
    save_image(OUTPUT_DIR / "rect_mask.png", rect_mask.astype(np.uint8) * 255)
    save_image(OUTPUT_DIR / "circle_noisy.png", noisy)
    save_image(OUTPUT_DIR / "circle_on_gradient.jpg", circle_image,
               jpeg_quality=95)
    save_image(OUTPUT_DIR / "circle_on_gradient.tif", circle_image)

    for name in (
        "circle_on_gradient.png",
        "circle_mask.png",
        "rect_on_flat.png",
        "rect_mask.png",
        "circle_noisy.png",
        "circle_on_gradient.jpg",
        "circle_on_gradient.tif",
    ):
        path = OUTPUT_DIR / name
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
