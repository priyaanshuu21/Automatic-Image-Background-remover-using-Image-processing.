# Automatic Image Background Removal Using Image Processing Techniques

A classical, dependency-light image processing project that removes the
background of a photograph automatically. The background is assumed to be the
region touching the image border; it is modelled statistically in the CIELAB
colour space, segmented with Otsu thresholding plus edge-barrier-aware region
growing, cleaned with binary morphology and exported as an RGBA image with a
transparent background. Every operator is implemented from scratch with NumPy
and is verified against OpenCV as a reference oracle in the test suite.

Status: under construction

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Lint

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

## Layout

- `bgremover/` - the library package (configuration, I/O, fundamentals, pixel
  relationships, point processing, filters, histograms, colour spaces,
  segmentation, morphology, edge detection, refinement, pipeline).
- `tests/` - headless pytest suite built on synthetic images only.
- `scripts/` - helper scripts (sample generation, verification, demo).
- `docs/` - progress log and syllabus map.
