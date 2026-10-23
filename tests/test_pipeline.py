import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import torch

import pipeline
from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from features import extract_slide, validate_features
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_reader import SdpcSlide, OpenSlideAdapter


class FakeSlide:
    base_mpp = 0.206
    level_dimensions = ((4096, 4096), (2048, 2048), (1024, 1024))
    level_downsamples = (1.0, 2.0, 4.0)

    def get_best_level_for_downsample(self, value):
        return min(range(3), key=lambda level: abs(self.level_downsamples[level] - value))

    def read_region(self, location, level, size):
        return Image.new("RGB", size, (180, 70, 120))

    def close(self):
        pass


class FakeEncoder(torch.nn.Module):
    def forward(self, images):
        return images.mean((1, 2, 3)).unsqueeze(1).expand(-1, 1024)


def transform(image):
    return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_coords(self):
        path = self.root / "sample.h5"
        with h5py.File(path, "w") as out:
            coords = out.create_dataset("coords", data=np.array([[0, 0], [100, 100]], dtype=np.int64))
            coords.attrs.update(patch_size=224, patch_level=1, source_level=1,
                                source_read_size=272, output_patch_size=224, target_mpp=0.5)
        return path

    def test_mpp_geometry_and_legacy_dataset(self):
        with patch("wsi_core.WholeSlideImage.open_wsi", return_value=FakeSlide()):
            wsi = WholeSlideImage("sample.sdpc")
        contour = np.array([[[0, 0]], [[4000, 0]], [[4000, 4000]], [[0, 4000]]], dtype=np.int32)
        assets, metadata = wsi.process_contour(contour, [], 0, str(self.root),
                                               patch_size=224, step_size=224,
                                               contour_fn="center", target_mpp=0.5)
        attrs = metadata["coords"]
        self.assertEqual(attrs["source_level"], 1)
        self.assertEqual(attrs["source_read_size"], 272)
        self.assertLessEqual(abs(attrs["level0_patch_size"] * 0.206 - 112), 0.206)
        self.assertGreater(len(assets["coords"]), 0)
        path = self.make_coords()
        dataset = Whole_Slide_Bag_FP(path, wsi=FakeSlide(), img_transforms=transform)
        self.assertEqual(tuple(dataset[0]["img"].shape), (3, 224, 224))
        dataset.close()
        with h5py.File(path, "a") as out:
            for key in ("source_level", "source_read_size", "output_patch_size"):
                del out["coords"].attrs[key]
        dataset = Whole_Slide_Bag_FP(path, wsi=FakeSlide(), img_transforms=transform)
        self.assertEqual(dataset.patch_size, 224)
        dataset.close()

    def test_empty_tissue(self):
        fake = FakeSlide()
        fake.read_region = lambda location, level, size: Image.new("RGB", size, "white")
        with patch("wsi_core.WholeSlideImage.open_wsi", return_value=fake):
            wsi = WholeSlideImage("empty.sdpc")
        wsi.segmentTissue(seg_level=2, filter_params={"a_t": 100, "a_h": 8, "max_n_holes": 32})
        self.assertEqual(wsi.contours_tissue, [])
        self.assertIsNone(wsi.process_contours(str(self.root)))

    def test_sdpc_white_padding(self):
        adapter = SdpcSlide.__new__(SdpcSlide)
        adapter._slide = FakeSlide()
        adapter.level_count = 3
        adapter.level_dimensions = FakeSlide.level_dimensions
        adapter.level_downsamples = FakeSlide.level_downsamples
        image = np.asarray(adapter.read_region((-4, -4), 1, (8, 8)))
        self.assertTrue((image[:2] == 255).all())
        self.assertTrue((image[:, :2] == 255).all())
        self.assertTrue((image[2:, 2:] == [180, 70, 120]).all())
        self.assertTrue((np.asarray(adapter.read_region((5000, 5000), 0, (8, 8))) == 255).all())
        adapter.close()

    def test_feature_outputs_resume_and_corruption(self):
        coords = self.make_coords()
        slide = self.root / "sample.sdpc"
        with patch("wsi_core.wsi_reader.open_wsi", return_value=FakeSlide()):
            count = extract_slide(slide, coords, self.root / "features", FakeEncoder(), transform,
                                  model_name="uni_v1", signature="test", device=torch.device("cpu"))
        self.assertEqual(count, 2)
        h5 = self.root / "features/h5_files/sample.h5"
        pt = self.root / "features/pt_files/sample.pt"
        self.assertEqual(validate_features(h5, pt, coords, 1024, "test"), 2)
        self.assertEqual(extract_slide(slide, coords, self.root / "features", None, transform,
                                       model_name="uni_v1", signature="test",
                                       device=torch.device("cpu")), 2)
        with self.assertRaises(ValueError):
            validate_features(h5, pt, coords, 1024, "different-model")
        with h5py.File(h5, "a") as out:
            out["features"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_features(h5, pt, coords, 1024, "test")

    def test_preprocess_end_to_end_and_resume(self):
        source = self.root / "slides"
        source.mkdir()
        (source / "sample.sdpc").touch()
        preset = json.loads((Path(pipeline.__file__).parent / "presets/balanced.json").read_text())
        preset["filter"]["a_t"] = 1
        preset_path = self.root / "small_tissue.json"
        preset_path.write_text(json.dumps(preset))
        args = ["--source", str(source), "--output", str(self.root / "output"),
                "--stage", "preprocess", "--target-mpp", "0.5", "--stitch",
                "--preset", str(preset_path)]
        with patch("wsi_core.WholeSlideImage.open_wsi", return_value=FakeSlide()):
            self.assertEqual(pipeline.main(args), 0)
        self.assertTrue((self.root / "output/masks/sample.png").is_file())
        self.assertTrue((self.root / "output/stitches/sample.png").is_file())
        with patch("wsi_core.WholeSlideImage.open_wsi", side_effect=AssertionError("should skip")):
            self.assertEqual(pipeline.main(args), 0)
        with self.assertRaises(ValueError):
            pipeline.main(args + ["--target-mpp", "1.0"])

    def test_discovery_and_exclusions(self):
        for name in ("a.sdpc", "b.SDPC", "c.tif", "weights.bin"):
            (self.root / name).touch()
        exclude = self.root / "exclude.txt"
        exclude.write_text("a\n", encoding="utf-8")
        args = pipeline.parse_args(["--source", str(self.root), "--output", str(self.root / "out"),
                                    "--stage", "preprocess", "--exclude", str(exclude)])
        self.assertEqual([p.name for p in pipeline.discover(args)], ["b.SDPC", "c.tif"])

    def test_openslide_native_and_virtual(self):
        fake = FakeSlide()
        fake.properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}
        with patch("wsi_core.wsi_reader.openslide.open_slide", return_value=fake):
            native = OpenSlideAdapter("sample.svs")
            self.assertEqual(native.level_count, 3)
            self.assertEqual(native.read_region((0, 0), 1, (12, 12)).size, (12, 12))
        fake.level_dimensions = ((4096, 4096),)
        fake.level_downsamples = (1.0,)
        calls = []
        def read(location, level, size):
            calls.append(size)
            return Image.new("RGBA", size, (0, 0, 0, 0))
        fake.read_region = read
        with patch("wsi_core.wsi_reader.openslide.open_slide", return_value=fake):
            virtual = OpenSlideAdapter("single_level.tif", max_source_edge=32)
            image = virtual.read_region((0, 0), 1, (32, 32))
            self.assertTrue((np.array(image) == 255).all())
            self.assertTrue(all(max(size) <= 32 for size in calls))


if __name__ == "__main__":
    unittest.main()
