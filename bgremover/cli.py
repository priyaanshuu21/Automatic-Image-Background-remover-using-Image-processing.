"""Command-line interface for background removal.

Subcommands:

* ``remove`` — remove the background of one image file,
* ``batch`` — remove the background of every matching file in a folder,
* ``stages`` — save the ten intermediate stage grids of one image,
* ``analyze`` — save a frequency-transform figure of one image,
* ``info`` — print the version and the default configuration.

Every command returns an exit code and never opens a window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from bgremover import __version__
from bgremover.config import PipelineConfig
from bgremover.io import load_image
from bgremover.pipeline import BackgroundRemovalPipeline

__all__ = ["build_parser", "main"]


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by every pipeline-running subcommand."""
    parser.add_argument(
        "--denoise",
        choices=("gaussian", "median", "none"),
        default="gaussian",
        help="pre-filter applied before segmentation",
    )
    parser.add_argument(
        "--work-max-side",
        type=int,
        default=800,
        help="longest side of the working resolution",
    )
    parser.add_argument(
        "--no-edge-barrier",
        action="store_true",
        help="let region growing cross Canny edges",
    )
    parser.add_argument(
        "--keep-largest",
        type=int,
        default=1,
        help="number of largest components kept (0 keeps all)",
    )
    parser.add_argument(
        "--background-modes",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of background colour clusters",
    )


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a validated configuration from parsed arguments."""
    return PipelineConfig(
        denoise=args.denoise,
        work_max_side=args.work_max_side,
        use_edge_barrier=not args.no_edge_barrier,
        keep_largest=args.keep_largest,
        n_background_modes=args.background_modes,
    )


def _command_remove(args: argparse.Namespace) -> int:
    """Run the ``remove`` subcommand."""
    try:
        config = _config_from_args(args)
        result = BackgroundRemovalPipeline(config).run_file(
            args.input, args.output
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    foreground = 100.0 * float(result.mask.mean())
    print(f"wrote {args.output} "
          f"(foreground {foreground:.1f}%, scale {result.scale:.3f})")
    return 0


def _command_batch(args: argparse.Namespace) -> int:
    """Run the ``batch`` subcommand."""
    source = Path(args.input_dir)
    destination = Path(args.output_dir)
    if not source.is_dir():
        print(f"error: not a directory: {source}", file=sys.stderr)
        return 1
    files = sorted(source.glob(args.pattern))
    if not files:
        print(f"error: no files match {args.pattern} in {source}",
              file=sys.stderr)
        return 1
    try:
        config = _config_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    pipeline = BackgroundRemovalPipeline(config)
    failures = 0
    for path in files:
        target = destination / f"{path.stem}_rgba.png"
        try:
            result = pipeline.run_file(str(path), str(target))
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        foreground = 100.0 * float(result.mask.mean())
        print(f"wrote {target} (foreground {foreground:.1f}%)")
    done = len(files) - failures
    print(f"batch done: {done} succeeded, {failures} failed")
    return 1 if failures else 0


def _command_stages(args: argparse.Namespace) -> int:
    """Run the ``stages`` subcommand."""
    from bgremover.viz import save_stage_grid

    try:
        config = _config_from_args(args)
        result = BackgroundRemovalPipeline(config).run(
            load_image(args.input)
        )
        saved = save_stage_grid(args.grid, result.stages)
        if args.rgba is not None:
            from bgremover.io import save_image

            save_image(args.rgba, result.rgba)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {saved} with {len(result.stages)} stages")
    return 0


def _command_analyze(args: argparse.Namespace) -> int:
    """Run the ``analyze`` subcommand."""
    import matplotlib.pyplot as plt

    from bgremover.fundamentals import to_gray
    from bgremover.transforms import (
        dct2,
        dft2,
        magnitude_spectrum,
    )

    try:
        gray = to_gray(load_image(args.input))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    spectrum = magnitude_spectrum(dft2(gray))
    coefficients = np.abs(dct2(gray))
    energy = float((coefficients ** 2).sum())
    quarter_h, quarter_w = gray.shape[0] // 4, gray.shape[1] // 4
    low = coefficients[:quarter_h, :quarter_w]
    share = float((low ** 2).sum() / energy) if energy > 0 else 0.0
    display = np.log1p(coefficients)
    peak = float(display.max())
    display = display / peak if peak > 0 else display
    figure, axes = plt.subplots(1, 3)
    figure.set_size_inches(9.6, 3.2)
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("gray input")
    axes[1].imshow(spectrum, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("DFT magnitude")
    axes[2].imshow(display, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("DCT-II log coefficients")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100)
    plt.close(figure)
    print(f"wrote {output} "
          f"(low-frequency DCT energy share: {share:.3f})")
    return 0


def _command_info(_: argparse.Namespace) -> int:
    """Run the ``info`` subcommand."""
    print(f"bgremover {__version__}")
    for name in PipelineConfig().__dataclass_fields__:
        print(f"{name}={getattr(PipelineConfig(), name)!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="bgremover",
        description="Classical automatic image background removal.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    remove = commands.add_parser("remove", help="process one image")
    remove.add_argument("input", help="source image file")
    remove.add_argument("output", help="destination RGBA file (.png)")
    _add_config_arguments(remove)
    remove.set_defaults(func=_command_remove)

    batch = commands.add_parser("batch", help="process a folder")
    batch.add_argument("input_dir", help="folder with source images")
    batch.add_argument("output_dir", help="folder receiving RGBA files")
    batch.add_argument("--pattern", default="*.png",
                       help="filename glob (default: *.png)")
    _add_config_arguments(batch)
    batch.set_defaults(func=_command_batch)

    stages = commands.add_parser(
        "stages", help="save the intermediate stage grid"
    )
    stages.add_argument("input", help="source image file")
    stages.add_argument("grid", help="destination figure file (.png)")
    stages.add_argument("--rgba", default=None,
                        help="optional destination RGBA file")
    _add_config_arguments(stages)
    stages.set_defaults(func=_command_stages)

    analyze = commands.add_parser(
        "analyze", help="save a frequency-transform figure"
    )
    analyze.add_argument("input", help="source image file")
    analyze.add_argument("output", help="destination figure file (.png)")
    analyze.set_defaults(func=_command_analyze)

    info = commands.add_parser("info", help="print version and config")
    info.set_defaults(func=_command_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m`` and the tests.

    Parameters
    ----------
    argv:
        Argument list without the program name; ``sys.argv[1:]`` when
        ``None``.

    Returns
    -------
    int
        The exit code of the selected subcommand.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
