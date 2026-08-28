from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
from PIL import Image

from detr.datasets.voc import VocDetection
from detr.validate import evaluate_voc


class VocDetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "VOC2007"
        (self.root / "Annotations").mkdir(parents=True)
        (self.root / "JPEGImages").mkdir()
        (self.root / "ImageSets" / "Main").mkdir(parents=True)

        image_id = "000001"
        Image.new("RGB", (20, 20), color="white").save(
            self.root / "JPEGImages" / f"{image_id}.jpg"
        )
        annotation = textwrap.dedent(
            """
            <annotation>
              <filename>000001.jpg</filename>
              <size><width>20</width><height>20</height><depth>3</depth></size>
              <object>
                <name>aeroplane</name><difficult>0</difficult>
                <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
              </object>
              <object>
                <name>aeroplane</name><difficult>1</difficult>
                <bndbox><xmin>11</xmin><ymin>11</ymin><xmax>20</xmax><ymax>20</ymax></bndbox>
              </object>
            </annotation>
            """
        ).strip()
        (self.root / "Annotations" / f"{image_id}.xml").write_text(annotation)
        for image_set in ("train", "val"):
            (self.root / "ImageSets" / "Main" / f"{image_set}.txt").write_text(
                f"{image_id}\n"
            )

        self.dataset = VocDetection(self.root, "train")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_parses_coordinates_labels_area_and_difficult(self) -> None:
        ground_truth = self.dataset.get_ground_truth("000001")
        torch.testing.assert_close(
            ground_truth["boxes"],
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]]),
        )
        self.assertEqual(ground_truth["labels"].tolist(), [0, 0])
        self.assertEqual(ground_truth["area"].tolist(), [100.0, 100.0])
        self.assertEqual(ground_truth["difficult"].tolist(), [False, True])

        _, training_target = self.dataset[0]
        self.assertEqual(training_target["boxes"].shape, (1, 4))
        self.assertEqual(training_target["labels"].tolist(), [0])

    def _detection(self, bbox: list[float], score: float) -> dict:
        return {
            "image_id": "000001",
            "label": 0,
            "bbox": bbox,
            "score": score,
        }

    def test_perfect_detection_has_full_ap(self) -> None:
        metrics = evaluate_voc(
            [self._detection([0.0, 0.0, 10.0, 10.0], 0.9)],
            self.dataset,
        )
        self.assertAlmostEqual(metrics["per_class_ap"]["aeroplane"], 1.0)
        self.assertAlmostEqual(metrics["map"], 1.0)

    def test_false_positive_before_true_positive_reduces_ap(self) -> None:
        metrics = evaluate_voc(
            [
                self._detection([0.0, 10.0, 5.0, 15.0], 0.9),
                self._detection([0.0, 0.0, 10.0, 10.0], 0.8),
            ],
            self.dataset,
        )
        self.assertAlmostEqual(metrics["per_class_ap"]["aeroplane"], 0.5)

    def test_difficult_match_is_ignored(self) -> None:
        metrics = evaluate_voc(
            [
                self._detection([10.0, 10.0, 20.0, 20.0], 0.95),
                self._detection([0.0, 0.0, 10.0, 10.0], 0.9),
            ],
            self.dataset,
        )
        self.assertAlmostEqual(metrics["per_class_ap"]["aeroplane"], 1.0)

    def test_duplicate_detection_does_not_create_another_true_positive(self) -> None:
        metrics = evaluate_voc(
            [
                self._detection([0.0, 0.0, 10.0, 10.0], 0.9),
                self._detection([0.0, 0.0, 10.0, 10.0], 0.8),
            ],
            self.dataset,
        )
        self.assertAlmostEqual(metrics["per_class_ap"]["aeroplane"], 1.0)


if __name__ == "__main__":
    unittest.main()
