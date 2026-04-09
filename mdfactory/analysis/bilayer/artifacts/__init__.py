"""Bilayer artifact generators using VMD TCL scripts."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdfactory.analysis.simulation import Simulation


def _require_vmd() -> str:
    vmd_bin = shutil.which("vmd")
    if not vmd_bin:
        raise FileNotFoundError("VMD executable not found on PATH (expected `vmd`).")
    return vmd_bin


def _require_ffmpeg() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError("ffmpeg executable not found on PATH (expected `ffmpeg`).")
    return ffmpeg_bin


def _artifact_dir(simulation: "Simulation") -> Path:
    artifact_dir = simulation.analysis_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def render_bilayer_snapshot(
    simulation: "Simulation",
    *,
    output_prefix: str = "bilayer_snapshot",
) -> list[Path]:
    """Render snapshot images of the last trajectory frame."""
    vmd_bin = _require_vmd()
    artifact_dir = _artifact_dir(simulation)
    tcl_script = Path(__file__).with_name("bilayer_render_script.tcl")

    prefix_path = artifact_dir / output_prefix
    cmd = [
        vmd_bin,
        "-dispdev",
        "text",
        "-e",
        str(tcl_script),
        "-args",
        str(simulation.structure_file),
        str(simulation.trajectory_file),
        str(prefix_path),
    ]
    subprocess.run(cmd, check=True)

    tga_outputs = [
        prefix_path.with_name(f"{prefix_path.name}_top.tga"),
        prefix_path.with_name(f"{prefix_path.name}_side.tga"),
    ]
    png_outputs = [path.with_suffix(".png") for path in tga_outputs]

    from PIL import Image

    for tga_path, png_path in zip(tga_outputs, png_outputs, strict=False):
        if not tga_path.exists():
            raise FileNotFoundError(f"Snapshot output not found: {tga_path}")
        with Image.open(tga_path) as image:
            image.save(png_path)

    for tga_path in prefix_path.parent.glob(f"{prefix_path.name}_*.tga"):
        tga_path.unlink(missing_ok=True)

    destination_dir = prefix_path.parent / output_prefix
    if destination_dir.is_dir():
        for tga_path in destination_dir.glob(f"{prefix_path.name}_*.tga"):
            tga_path.unlink(missing_ok=True)

    return png_outputs


def render_bilayer_movie(
    simulation: "Simulation",
    *,
    output_prefix: str = "bilayer_movie",
    framerate: int = 15,
    crf: int = 18,
    preset: str = "slow",
    frame_step: int = 2,
    max_frames: int | None = None,
) -> list[Path]:
    """Render a bilayer movie and encode to MP4 with ffmpeg."""
    vmd_bin = _require_vmd()
    ffmpeg_bin = _require_ffmpeg()
    artifact_dir = _artifact_dir(simulation)
    tcl_script = Path(__file__).with_name("render_movie_smooth.tcl")

    prefix_path = artifact_dir / output_prefix
    cmd = [
        vmd_bin,
        "-dispdev",
        "text",
        "-e",
        str(tcl_script),
        "-args",
        str(simulation.structure_file),
        str(simulation.trajectory_file),
        str(prefix_path),
        str(frame_step),
        str(max_frames or 0),
    ]
    subprocess.run(cmd, check=True)

    frames_dir = prefix_path.with_name(f"{prefix_path.name}_frames")
    movie_path = prefix_path.with_name(f"{prefix_path.name}_movie.mp4")

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(framerate),
        "-i",
        str(frames_dir / "frame_%05d.tga"),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(movie_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    return [movie_path]
