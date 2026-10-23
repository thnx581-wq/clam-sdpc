"""Batched extraction with per-worker readers and atomic slide-level outputs."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from models.builder import FEATURE_DIMS


def validate_features(h5_path, pt_path, coords_path, dimension, signature):
    """Validate order, count, provenance and finite values before resume/commit."""
    with h5py.File(coords_path, "r") as source, h5py.File(h5_path, "r") as result:
        count = len(source["coords"])
        if count == 0 or result["features"].shape != (count, dimension):
            raise ValueError("Empty or mismatched feature shape")
        if result["coords"].shape != (count, 2):
            raise ValueError("Mismatched coordinate shape")
        if result.attrs.get("encoder_signature") != signature:
            raise ValueError("Encoder settings differ from existing features")
        for start in range(0, count, 4096):
            stop = min(count, start + 4096)
            if not np.array_equal(source["coords"][start:stop], result["coords"][start:stop]):
                raise ValueError("Feature coordinate order differs from patch coordinates")
            if not np.isfinite(result["features"][start:stop]).all():
                raise ValueError("Non-finite HDF5 features")
    tensor = torch.load(pt_path, map_location="cpu", weights_only=True, mmap=True)
    if tuple(tensor.shape) != (count, dimension) or not torch.isfinite(tensor).all():
        raise ValueError("Invalid PT features")
    with h5py.File(h5_path, "r") as result:
        for start in range(0, count, 4096):
            if not np.array_equal(tensor[start:start + 4096].numpy(), result["features"][start:start + 4096]):
                raise ValueError("HDF5 and PT features differ")
    return count


def extract_slide(slide, coords_path, feature_dir, model, transform, *,
                  model_name, signature, device, batch_size=64, workers=0, amp=False):
    feature_dir = Path(feature_dir)
    h5_path = feature_dir / "h5_files" / (slide.stem + ".h5")
    pt_path = feature_dir / "pt_files" / (slide.stem + ".pt")
    for path in (h5_path, pt_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    dimension = FEATURE_DIMS[model_name]
    if h5_path.exists() and pt_path.exists():
        return validate_features(h5_path, pt_path, coords_path, dimension, signature)
    dataset = Whole_Slide_Bag_FP(str(coords_path), wsi_path=str(slide), img_transforms=transform)
    if len(dataset) == 0:
        raise ValueError("No patch coordinates")
    loader_args = dict(batch_size=batch_size, num_workers=workers,
                       pin_memory=device.type == "cuda")
    if workers:
        loader_args.update(prefetch_factor=1, persistent_workers=False,
                           multiprocessing_context="spawn")
    h5_temp, pt_temp = h5_path.with_suffix(".h5.tmp"), pt_path.with_suffix(".pt.tmp")
    loader = DataLoader(dataset, **loader_args)
    try:
        with h5py.File(coords_path, "r") as source, h5py.File(h5_temp, "w") as out:
            count = len(dataset)
            features = out.create_dataset("features", (count, dimension), dtype="float32")
            coords = out.create_dataset("coords", data=source["coords"][:])
            for key, value in source["coords"].attrs.items():
                coords.attrs[key] = value
            out.attrs["encoder_signature"] = signature
            out.attrs["model_name"] = model_name
            offset = 0
            with torch.inference_mode():
                for batch in tqdm(loader, desc=slide.name):
                    images = batch["img"].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=amp and device.type == "cuda"):
                        values = model(images).float().cpu().numpy()
                    if values.shape != (len(images), dimension) or not np.isfinite(values).all():
                        raise ValueError(f"Invalid model output: {values.shape}")
                    if not np.array_equal(batch["coord"].numpy(), coords[offset:offset + len(images)]):
                        raise ValueError("DataLoader changed coordinate order")
                    features[offset:offset + len(images)] = values
                    offset += len(images)
            if offset != count:
                raise ValueError("Incomplete feature extraction")
        with h5py.File(h5_temp, "r") as out:
            torch.save(torch.from_numpy(out["features"][:]), pt_temp)
        validate_features(h5_temp, pt_temp, coords_path, dimension, signature)
        h5_temp.replace(h5_path)
        pt_temp.replace(pt_path)
        return count
    finally:
        dataset.close()
        del loader


def encoder_signature(name, checkpoint, amp):
    info = {"model": name, "input_size": 224, "amp": amp}
    if checkpoint:
        path = Path(checkpoint).resolve()
        info.update(checkpoint=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    return json.dumps(info, sort_keys=True)
