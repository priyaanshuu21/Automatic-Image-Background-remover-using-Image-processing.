# Project Progress

| Task | Status | Date | Notes |
|------|--------|------|-------|
| 00 | Done | 2026-10-02 | Workspace inspection, virtual environment, tooling config, git remote. `ruff check .` clean, `pytest -q` = 1 passed. |
| 01 | Done | 2026-10-02 | `config.py` (validated frozen `PipelineConfig`), `io.py` (Pillow I/O), `fundamentals.py` (gray conversion, channels, resampling, quantization, resizing), synthetic fixtures, `make_samples.py`, `verify.ps1`, architecture guard. `pytest -q` = 106 passed. |

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
