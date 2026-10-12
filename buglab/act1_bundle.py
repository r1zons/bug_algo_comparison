"""Export and inspect the polished Act I lecture asset bundle."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import shutil

from matplotlib import pyplot as plt
from PIL import Image

from .crossover_slides import export_crossover_slides
from .scenes import SceneConfig, export_manifest, load_scene_configs


def _gif_frame(path: Path, index: int) -> Image.Image:
    with Image.open(path) as image:
        image.seek(max(0, min(index, image.n_frames - 1)))
        return image.convert("RGB").copy()


def export_contact_sheet(configs: tuple[SceneConfig, ...], output_dir: Path) -> tuple[Path, Path]:
    """Create first/middle/final/message rows from already-rendered GIFs."""
    rows = len(configs)
    figure, axes = plt.subplots(rows, 4, figsize=(16, 3.1 * rows), facecolor="white")
    headers = ("First frame", "Middle frame", "Final frame", "Educational message")
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=15, fontweight="bold", pad=12)
    inspections: dict[str, dict[str, int]] = {}
    for row, config in enumerate(configs):
        gif_path = output_dir / f"{config.name}.gif"
        with Image.open(gif_path) as image:
            indices = (0, image.n_frames // 2, image.n_frames - 1)
            inspections[config.name] = {
                "first": indices[0], "middle": indices[1], "final": indices[2],
                "frame_count": image.n_frames,
            }
        for column, index in enumerate(indices):
            axes[row, column].imshow(_gif_frame(gif_path, index))
            axes[row, column].axis("off")
        message_ax = axes[row, 3]
        message_ax.axis("off")
        message_ax.text(.02, .72, config.name.replace("_", " "), fontsize=13,
                        fontweight="bold", va="top")
        message_ax.text(.02, .52, config.educational_message, fontsize=12, va="top",
                        wrap=True, linespacing=1.35)
        message_ax.text(.02, .08, f"{config.fps} fps · {config.duration_seconds:g} s",
                        fontsize=10, color="#68737d")
    figure.suptitle("Act I visual inspection · recorded first, middle, and final frames",
                    fontsize=20, fontweight="bold", y=.997)
    figure.tight_layout(rect=(0, 0, 1, .985), h_pad=1.1, w_pad=.6)
    png_path = output_dir / "ACT1_CONTACT_SHEET.png"
    json_path = output_dir / "ACT1_CONTACT_SHEET.json"
    figure.savefig(png_path, dpi=120, facecolor="white")
    plt.close(figure)
    payload = {
        "schema_version": 1,
        "source": "already-rendered GIF frames; no optimizer or simulation launched",
        "inspected_frames": inspections,
        "artifact_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        "rendering_launches_optimizer": False,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return png_path, json_path


def preserve_optional_synthetic_gif(output_dir: Path,
                                    historical_dir: Path = Path("outputs/act1_before_polish")) -> None:
    """Copy the historical synthetic animation into an explicitly optional folder."""
    optional = output_dir / "optional"
    optional.mkdir(parents=True, exist_ok=True)
    for suffix in ("gif", "json"):
        source = historical_dir / f"04_crossover_mutation.{suffix}"
        if source.exists():
            shutil.copy2(source, optional / source.name)


def copy_bundle_documents(output_dir: Path) -> None:
    """Place the reviewed index and Russian notes beside the final assets."""
    documents = {
        Path("ACT1_ANIMATION_INDEX.md"): output_dir / "INDEX.md",
        Path("ACT1_PRESENTER_NOTES_RU.md"): output_dir / "PRESENTER_NOTES_RU.md",
    }
    for source, destination in documents.items():
        if not source.exists():
            raise FileNotFoundError(f"Required bundle document is missing: {source}")
        shutil.copy2(source, destination)


def export_final_bundle(manifest: Path, output_dir: Path, *, render: bool = True) -> None:
    """Build final assets strictly from saved playback and operation logs."""
    configs = load_scene_configs(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    if render:
        export_manifest(configs, output_dir, source_root=manifest.parent)
        export_crossover_slides(output_dir)
    preserve_optional_synthetic_gif(output_dir)
    copy_bundle_documents(output_dir)
    export_contact_sheet(configs, output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Export the polished Act I lecture package")
    parser.add_argument("--manifest", type=Path, default=Path("act1_final_scenes.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/act1_final"))
    parser.add_argument("--contact-sheet-only", action="store_true",
                        help="reuse existing GIFs and only regenerate inspection materials")
    args = parser.parse_args(argv)
    export_final_bundle(args.manifest, args.output, render=not args.contact_sheet_only)
    print(f"Act I package: {args.output.resolve()}")


if __name__ == "__main__":
    main()
