"""Tests for the command-line interface (in-process, headless)."""

from __future__ import annotations

import pytest

from bgremover import __version__
from bgremover.cli import build_parser, main
from bgremover.io import save_image


def _write(path, image) -> str:
    """Save a fixture image and return its path as a string."""
    save_image(path, image)
    return str(path)


def test_remove_command(rect_image, tmp_path, capsys) -> None:
    """``remove`` writes an RGBA file and reports the foreground."""
    source = _write(tmp_path / "in.png", rect_image)
    target = str(tmp_path / "out.png")
    assert main(["remove", source, target]) == 0
    message = capsys.readouterr().out
    assert "foreground" in message and target in message
    assert main(["remove", str(tmp_path / "missing.png"), target]) == 1


def test_batch_command(rect_image, circle_image, tmp_path) -> None:
    """``batch`` converts every match and summarises the run."""
    _write(tmp_path / "first.png", rect_image)
    _write(tmp_path / "second.png", circle_image)
    out = tmp_path / "batch_out"
    assert main(["batch", str(tmp_path), str(out)]) == 0
    assert (out / "first_rgba.png").is_file()
    assert (out / "second_rgba.png").is_file()
    assert main(["batch", str(tmp_path / "nope"), str(out)]) == 1
    assert main(["batch", str(out), str(out / "again"),
                 "--pattern", "*.tif"]) == 1


def test_stages_command(rect_image, tmp_path) -> None:
    """``stages`` saves the ten-stage grid and optionally the RGBA."""
    source = _write(tmp_path / "in.png", rect_image)
    grid = tmp_path / "grid.png"
    assert main(["stages", source, str(grid)]) == 0
    assert grid.is_file() and grid.stat().st_size > 0
    rgba = tmp_path / "extra.png"
    assert main(["stages", source, str(grid), "--rgba",
                 str(rgba)]) == 0
    assert rgba.is_file()
    assert main(["stages", str(tmp_path / "missing.png"),
                 str(grid)]) == 1


def test_analyze_command(rect_image, tmp_path) -> None:
    """``analyze`` saves a spectrum figure and prints the energy."""
    source = _write(tmp_path / "in.png", rect_image)
    figure = tmp_path / "spectrum.png"
    assert main(["analyze", source, str(figure)]) == 0
    assert figure.is_file() and figure.stat().st_size > 0
    assert main(["analyze", str(tmp_path / "missing.png"),
                 str(figure)]) == 1


def test_info_command(capsys) -> None:
    """``info`` prints the version and every default."""
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert __version__ in output
    assert "work_max_side" in output


def test_parser_rejects_bad_invocations() -> None:
    """Missing commands and bad flags exit with code 2."""
    parser = build_parser()
    assert parser.prog == "bgremover"
    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as bad_flag:
        main(["remove", "a.png", "b.png", "--denoise", "bogus"])
    assert bad_flag.value.code == 2


def test_config_flags_are_honoured(rect_image, tmp_path, capsys) -> None:
    """CLI flags reach the pipeline configuration."""
    source = _write(tmp_path / "in.png", rect_image)
    target = str(tmp_path / "out.png")
    assert main(["remove", source, target, "--denoise", "none",
                 "--no-edge-barrier", "--keep-largest", "0",
                 "--background-modes", "1"]) == 0
    assert "foreground" in capsys.readouterr().out
    assert main(["remove", source, target, "--work-max-side",
                 "1"]) == 1
