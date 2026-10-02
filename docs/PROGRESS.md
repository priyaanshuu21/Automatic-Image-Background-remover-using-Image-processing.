# Project Progress

| Task | Status | Date | Notes |
|------|--------|------|-------|
| 00 | Done | 2026-10-02 | Workspace inspection, virtual environment, tooling config, git remote. `ruff check .` clean, `pytest -q` = 1 passed. |
| 01 | Done | 2026-10-02 | `config.py` (validated frozen `PipelineConfig`), `io.py` (Pillow I/O), `fundamentals.py` (gray conversion, channels, resampling, quantization, resizing), synthetic fixtures, `make_samples.py`, `verify.ps1`, architecture guard. `pytest -q` = 106 passed. |
| 02 | Done | 2026-10-02 | `pixels.py` (neighbourhoods, D4/D8/euclidean distances, two-pass union-find labelling verified against `cv2.connectedComponents`, component stats, ASCII neighbourhood demo) and `point_ops.py` (LUT-based negative, stretch, threshold, slice, log, power law). `pytest -q` = 157 passed. |
| 03 | Done | 2026-10-02 | `filters.py` (padding, correlation/convolution, box and separable Gaussian kernels, median, unsharp, Laplacian, `denoise` dispatch) and `histogram.py` (histogram, CDF, equalisation, per-channel colour histograms, hue-preserving value equalisation, statistics). Gaussian and median filters verified against OpenCV. `pytest -q` = 206 passed. |
| 04 | Done | 2026-10-02 | `color.py` (RGB to HSV/HSI/CIELAB conversions, HSV to RGB, Lab to RGB, Euclidean/Delta E distance, wrap-around HSV slicing, deterministic trimmed-cluster border background model, normalised distance map, `border_mask`). HSV verified against `cv2.COLOR_RGB2HSV_FULL` and Lab against `cv2.COLOR_RGB2Lab`, both within one unit. `pytest -q` = 232 passed. |
| 05 | Done | 2026-10-02 | `segmentation.py` (Otsu thresholding, variance, lightness Otsu, region growing with reference/allowed/connectivity, hysteresis, region adjacency/means/merging). Matches OpenCV Otsu semantics for binary splits. `pytest -q` = 256 passed. |
| 06 | Done | 2026-10-02 | `morphology.py` (structuring elements, dilation/erosion, opening/closing, morphological gradient/top-hat/black-hat, hit-or-miss, thinning/thickening, mask cleanup: fill holes, remove small objects, remove border-touching objects). Classical implementations only. Dilate/erode verified exactly against OpenCV. Fixed the boolean `erode` polarity (`>= 0.5`) and required odd kernels. `pytest -q` = 260 passed. |
| 07 | Done | 2026-10-02 | `edges.py` (Sobel/Prewitt/Laplacian/LoG kernels, gradients, magnitude/direction, Sobel/Prewitt edges, Laplacian/LoG zero crossings, Canny with vectorised NMS/double-threshold/hysteresis, edge barrier, edge overlay). Sobel/Laplacian verified exactly against OpenCV; Canny runs in ~25 ms on fixtures. `pytest -q` = 296 passed. |
| 08 | Done | 2026-10-02 | `refine.py` (band-limited alpha feathering, BFS halo decontamination, RGBA compositing, mask upscaling), `pipeline.py` (`BackgroundRemovalPipeline`/`PipelineResult` with 10 stage keys: input/resized/denoised/distance/edges/seed_mask/grown_mask/cleaned_mask/alpha/rgba), `viz.py` (Agg checkerboard/stage-grid rendering), `scripts/run_demo.py` (IoU 1.0000 circle, 0.9958 rect). Barrier-blocked pixels fall back to the colour model. `pytest -q` = 326 passed. |
| 09 | Done | 2026-10-02 | `transforms.py` (2D DFT/IDFT, log magnitude spectrum, orthonormal DCT-II, Sylvester Hadamard; decoupled from the pipeline), `cli.py` (`remove`/`batch`/`stages`/`analyze`/`info` subcommands), `gui_controller.py` (headless app logic) + `gui.py` (thin Tk view, never opens windows in tests). `pytest -q` = 359 passed. |
| 10 | Done | 2026-10-02 | Release 1.0.0: completed `README.md`, `docs/SYLLABUS_MAP.md`, `docs/PIPELINE_STAGES.md`; bumped `bgremover.__version__` to 1.0.0; fresh-venv verification green; tagged `v1.0.0`. |
| 11 | Done | 2026-10-02 | Universal classical engine: 16-zone perimeter seeding with edge/variance/model verification (`color.py`), saliency prior with nearest-mode + optional chroma gate, fused Sobel+Canny energy barrier with gated Lab growing (`edges.py`, `segmentation.py`), trimap + topological solidifier (`morphology.py`), distance-weighted guided feathering with bounded decontamination (`refine.py`), rewired `pipeline.py` (raw-colour seed/localisation, chromatic shadow absorption). Studio 0.9995, portrait 0.8449, two-tone 0.9949, fixtures 1.0000. GUI stage list intact. `pytest -q` = 388 passed. |

## Environment notes (Task 00)

- Python: 3.13.1 (only interpreter available via `py -0p`; `>= 3.10`).
- Virtual environment: `.venv` (created from `C:\Program Files\Python313`).
- Installed: numpy 2.5.3, opencv-python-headless (cv2) 5.0.0, Pillow 12.3.0,
  matplotlib 3.11.2, pytest 9.1.1, ruff, pytest-cov.
- Tkinter: available (TkVersion 8.6).
- Git: 2.52.0.windows.1, branch `main`, placeholder identity
  `Priyanshu <priyaanshuu21@users.noreply.github.com>` set with
  `git config --local` (change it if you prefer your real name/address).
- Remote probe (`git ls-remote --heads origin`, 20 s timeout): **reachable**
  (exit code 0). The remote repository currently has **no branches**, so the
  first `git push -u origin main` in Task 10 will create `main` there.
- Remote URL is stored exactly as supplied, including the double dot before
  `.git`; it was not "corrected".
