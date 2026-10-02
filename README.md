# Automatic Image Background Removal Using Image Processing Techniques

A classical, dependency-light image processing project that removes the
background of a photograph automatically. The background is assumed to be the
region touching the image border: it is modelled statistically in the CIELAB
colour space, grown from border seeds (stopped by Canny edges when configured),
cleaned with binary morphology and exported as an RGBA image with a
transparent background. Every operator is implemented from scratch with NumPy
and is verified against OpenCV as a reference oracle in the test suite.

Status: released as **v1.0.0** — all ten implementation tasks are done
(359 tests passing) and the full pipeline reaches IoU 1.0000 / 0.9958 on the
synthetic fixtures.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Requires Python 3.10+. Runtime dependencies are NumPy, OpenCV-Python
(headless), Pillow and Matplotlib; test/lint tooling adds pytest, ruff and
pytest-cov (see `requirements-dev.txt`).

## Quick start

Remove the background of one photo:

```powershell
.\.venv\Scripts\python.exe -m bgremover.cli remove photo.jpg photo_rgba.png
```

(The CLI module is executed as `python -m bgremover.cli` — see
`bgremover/cli.py::main` for the `remove`, `batch`, `stages`, `analyze`
and `info` subcommands.)

Run the scripted demo on the synthetic fixtures:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
```

Launch the desktop front-end:

```powershell
.\.venv\Scripts\python.exe -c "from bgremover.gui import launch; launch()"
```

## How it works

1. The image is resized to a working resolution and denoised.
2. The border frame is sampled in CIELAB and reduced to one or two robust
   background colour modes (median + MAD trimming).
3. Every pixel's Lab distance to the background model forms a normalised
   distance map; pixels past the halfway mark seed the foreground estimate.
4. Canny edges (fractional double thresholds + hysteresis) become a barrier
   that border-seeded region growing cannot cross.
5. Barrier pixels the growth could not reach fall back to the colour model.
6. The foreground mask is cleaned (small objects, holes, largest components),
   feathered only inside a narrow boundary band, halo-decontaminated and
   composited to RGBA at the original resolution.

See `docs/PIPELINE_STAGES.md` for the ten recorded intermediate stages and
`docs/SYLLABUS_MAP.md` for how each image processing topic maps to a module.

## Tests and lint

```powershell
.\scripts\verify.ps1          # ruff check . && pytest -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

The suite is fully headless and deterministic: every fixture is generated
from closed-form formulas with fixed seeds, Matplotlib uses the Agg backend
and Tk windows are never created during tests.

## Layout

- `bgremover/` — the library package: `config`, `io`, `fundamentals`,
  `pixels`, `point_ops`, `filters`, `histogram`, `color`, `segmentation`,
  `morphology`, `edges`, `refine`, `pipeline`, `viz`, `transforms`, `cli`,
  `gui_controller`, `gui`.
- `tests/` — headless pytest suite built on synthetic images only.
- `scripts/` — helper scripts (sample generation, verification, demo).
- `docs/` — progress log, syllabus map and pipeline stage reference.
- `samples/` — a `.gitkeep` placeholder; generated PNGs go to the
  git-ignored `samples/generated/`.
