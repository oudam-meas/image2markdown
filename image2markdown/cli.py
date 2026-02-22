import argparse
from pathlib import Path
import sys
import yaml

from .ocr import process_single_image, DEFAULT_MODEL, DEFAULT_PROMPT


def load_config(config_path: Path | None):
    if config_path is None:
        return {}
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def cli():
    parser = argparse.ArgumentParser(
        description="Convert image(s) to Markdown using a local Ollama vision model."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (optional).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # single
    p_single = subparsers.add_parser("single", help="Process a single image")
    p_single.add_argument("image", type=str, help="Path to input image")
    p_single.add_argument("output", type=str, help="Path to output .md file")

    # batch
    p_batch = subparsers.add_parser("batch", help="Process all images in a folder")
    p_batch.add_argument("input_dir", type=str, help="Folder with images")
    p_batch.add_argument("output_dir", type=str, help="Folder for markdown output")
    p_batch.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="Glob pattern for images (default: *.jpg)",
    )

    args = parser.parse_args()

    cfg = load_config(Path(args.config)) if args.config else {}

    model = cfg.get("model", DEFAULT_MODEL)
    prompt = cfg.get("prompt", DEFAULT_PROMPT)

    try:
        if args.command == "single":
            image = Path(args.image)
            output = Path(args.output)
            process_single_image(image, output, model=model, prompt=prompt)
            print(f"✓ {image} -> {output}")

        elif args.command == "batch":
            in_dir = Path(args.input_dir).expanduser().resolve()
            out_dir = Path(args.output_dir).expanduser().resolve()

            if not in_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {in_dir}")

            out_dir.mkdir(parents=True, exist_ok=True)

            images = list(in_dir.rglob(args.pattern))
            if not images:
                print(f"No images found in {in_dir} matching {args.pattern}")
                return

            for img in images:
                rel = img.relative_to(in_dir)
                out_path = out_dir / rel.with_suffix(".md")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"Processing {img} -> {out_path}")
                process_single_image(img, out_path, model=model, prompt=prompt)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
