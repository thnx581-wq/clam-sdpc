"""Portable CLAM segmentation, coordinate generation and feature extraction."""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import h5py
import numpy as np

EXTENSIONS = {".sdpc", ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn"}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["segment", "preprocess", "features", "all"], default="all")
    parser.add_argument("--target-mpp", type=float, default=1.0)
    parser.add_argument("--patch-size", type=positive_int, default=224)
    parser.add_argument("--step-size", type=positive_int, help="default: patch size")
    parser.add_argument("--preset", type=Path, default=Path(__file__).parent / "presets/balanced.json")
    parser.add_argument("--slides", nargs="+", help="exact filenames in source (optional)")
    parser.add_argument("--exclude", type=Path, help="UTF-8 text, one filename or stem per line")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--stitch", action="store_true")
    parser.add_argument("--coords-dir", type=Path, help="features only: existing patches/ directory")
    parser.add_argument("--model", choices=["uni_v1", "h_optimus_0", "resnet50_trunc"], default="uni_v1")
    parser.add_argument("--checkpoint", type=Path, help="local .bin/.pth; required for UNI/H-optimus")
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--workers", type=int, default=0, help="0 recommended on Windows; try 1-3 if RAM permits")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true", help="CUDA float16 inference")
    args = parser.parse_args(argv)
    args.step_size = args.step_size or args.patch_size
    if not np.isfinite(args.target_mpp) or args.target_mpp <= 0:
        parser.error("--target-mpp must be finite and positive")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.coords_dir and args.stage != "features":
        parser.error("--coords-dir is only valid with --stage features")
    if args.stage in ("all", "features"):
        if args.model != "resnet50_trunc" and args.checkpoint is None:
            parser.error("--checkpoint is required for UNI and H-optimus")
        if args.checkpoint is not None and not args.checkpoint.is_file():
            parser.error("--checkpoint does not exist")
    return args


def discover(args):
    source = args.source.resolve(strict=True)
    available = {path.name: path for path in source.iterdir()
                 if path.is_file() and path.suffix.lower() in EXTENSIONS}
    excluded = set()
    if args.exclude:
        excluded = {line.strip() for line in args.exclude.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip() and not line.startswith("#")}
    names = args.slides if args.slides is not None else sorted(available)
    missing = set(names) - available.keys()
    if missing:
        raise ValueError(f"Unknown slide filename(s): {sorted(missing)}")
    slides = [available[name] for name in dict.fromkeys(names)
              if name not in excluded and Path(name).stem not in excluded]
    if args.limit:
        slides = slides[:args.limit]
    if not slides:
        raise ValueError("No matching slides")
    if len({path.stem.casefold() for path in slides}) != len(slides):
        raise ValueError("Duplicate slide stems would overwrite outputs; use separate output directories")
    return slides


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def validate_coords(path, args, preset=None):
    with h5py.File(path, "r") as handle:
        coords = handle["coords"]
        if coords.shape[1:] != (2,) or len(coords) == 0 or coords.dtype.kind not in "iu":
            raise ValueError("Empty or malformed coordinate array")
        for key, expected in (("target_mpp", args.target_mpp),
                              ("output_patch_size", args.patch_size)):
            if key not in coords.attrs or not np.isclose(float(coords.attrs[key]), expected):
                raise ValueError(f"Patch metadata mismatch: {key}; use matching settings/new output")
        if preset is not None and handle.attrs.get("preset") != json.dumps(preset, sort_keys=True):
            raise ValueError("Segmentation preset differs; use a new output directory")
        return len(coords)


def preprocess(slide, args, preset):
    from wsi_core.WholeSlideImage import WholeSlideImage
    from wsi_core.wsi_utils import StitchCoords

    root = args.output
    patch_path = root / "patches" / (slide.stem + ".h5")
    mask_path = root / "masks" / (slide.stem + ".png")
    seg_path = root / "contours" / (slide.stem + ".pkl")
    stitch_path = root / "stitches" / (slide.stem + ".png")
    if args.stage != "segment" and patch_path.exists():
        count = validate_coords(patch_path, args, preset)
        if mask_path.exists() and seg_path.exists() and (not args.stitch or stitch_path.exists()):
            return count
    wsi = WholeSlideImage(str(slide))
    try:
        seg = preset["segmentation"].copy()
        level = seg.pop("seg_level", -1)
        if level < 0:
            level = wsi.wsi.get_best_level_for_downsample(64)
        if np.prod(wsi.level_dim[level], dtype=np.int64) > 100_000_000:
            raise ValueError("Segmentation overview exceeds 100 million pixels; choose a lower resolution")
        wsi.segmentTissue(seg_level=level, filter_params=preset["filter"], **seg)
        wsi.visWSI(vis_level=level, line_thickness=250).save(mask_path)
        wsi.saveSegmentation(str(seg_path))
        if args.stage == "segment":
            return 0
        if not patch_path.exists():
            temp_dir = root / "patches" / ".incomplete"
            temp_dir.mkdir(exist_ok=True)
            path = wsi.process_contours(str(temp_dir), patch_size=args.patch_size,
                                       step_size=args.step_size, target_mpp=args.target_mpp,
                                       **preset["patch"])
            if path is None:
                raise ValueError("No tissue patches; inspect mask/preset")
            with h5py.File(path, "a") as handle:
                handle.attrs["preset"] = json.dumps(preset, sort_keys=True)
            validate_coords(path, args, preset)
            Path(path).replace(patch_path)
        if args.stitch and not stitch_path.exists():
            StitchCoords(str(patch_path), wsi, downscale=64).save(stitch_path)
        return validate_coords(patch_path, args, preset)
    finally:
        wsi.close()


def main(argv=None):
    args = parse_args(argv)
    slides = discover(args)
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("masks", "contours", "patches", "stitches", "logs"):
        (args.output / name).mkdir(exist_ok=True)
    config = {"source": str(args.source.resolve()), "target_mpp": args.target_mpp,
              "patch_size": args.patch_size, "step_size": args.step_size, "preset": preset}
    config_path = args.output / "preprocess_config.json"
    if args.stage != "features":
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing output has different settings; choose a new --output")
        write_json(config_path, config)
    file_handler = logging.FileHandler(args.output / "logs/pipeline.log", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), file_handler], force=True)
    model = None
    failed = 0
    with (args.output / "run_status.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["slide", "stage", "status", "patches", "error"])
        writer.writeheader()
        for slide in slides:
            row = dict(slide=slide.name, stage=args.stage, status="complete", patches=0, error="")
            try:
                if args.stage != "features":
                    row["patches"] = preprocess(slide, args, preset)
                if args.stage in ("all", "features"):
                    from features import extract_slide, encoder_signature
                    from models.builder import get_encoder
                    import torch
                    coords = (args.coords_dir or args.output / "patches") / (slide.stem + ".h5")
                    validate_coords(coords, args)
                    if model is None:
                        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                                              else "cpu" if args.device == "auto" else args.device)
                        model, transform = get_encoder(args.model, args.checkpoint)
                        model.to(device)
                    row["patches"] = extract_slide(
                        slide, coords, args.output / "features" / args.model, model, transform,
                        model_name=args.model, signature=encoder_signature(args.model, args.checkpoint, args.amp),
                        device=device, batch_size=args.batch_size, workers=args.workers, amp=args.amp)
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                failed += 1
                logging.exception("Failed %s", slide.name)
            writer.writerow(row)
            stream.flush()
            logging.info("%s", row)
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
