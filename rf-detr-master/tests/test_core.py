from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rfdetr_tutorial.backbone import (
    DINOv2Backbone,
    DINOv2WindowedLayer,
    MultiScaleProjector,
    TinyViTBackbone,
)
from rfdetr_tutorial.config import load_config
from rfdetr_tutorial.criterion import SetCriterion
from rfdetr_tutorial.model import build_model
from rfdetr_tutorial.nas import pareto_frontier


ROOT = Path(__file__).parents[1]


class CoreTest(unittest.TestCase):
    def test_projector_resamples_and_concatenates_every_backbone_feature(self) -> None:
        projector = MultiScaleProjector(input_channels=32, hidden_size=16, num_input_features=4)
        fusion_inputs: list[torch.Size] = []
        handles = [
            stage.register_forward_pre_hook(
                lambda module, inputs: fusion_inputs.append(inputs[0].shape)
            )
            for stage in projector.fusion_stages
        ]
        try:
            pyramid, masks, positions = projector(
                [torch.randn(2, 32, 8, 8) for _ in range(4)],
                torch.zeros(2, 8, 8, dtype=torch.bool),
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(fusion_inputs, [(2, 64, 16, 16), (2, 128, 8, 8), (2, 128, 4, 4)])
        self.assertEqual(
            [feature.shape for feature in pyramid],
            [(2, 16, 16, 16), (2, 16, 8, 8), (2, 16, 4, 4)],
        )
        self.assertEqual([mask.shape for mask in masks], [(2, 16, 16), (2, 8, 8), (2, 4, 4)])
        self.assertEqual([position.shape for position in positions], [feature.shape for feature in pyramid])

    def test_tiny_vit_switches_between_window_and_global_attention(self) -> None:
        backbone = TinyViTBackbone(
            hidden_size=192,
            patch_size=16,
            layers=4,
            num_windows=2,
            window_block_indexes=[0, 2],
        )
        sequence_lengths: list[list[int]] = [[] for _ in backbone.layers]
        handles = []
        for layer_index, layer in enumerate(backbone.layers):
            def record_length(module: nn.Module, inputs: tuple[torch.Tensor, ...], index: int = layer_index) -> None:
                sequence_lengths[index].append(inputs[0].shape[1])

            handles.append(layer.register_forward_pre_hook(record_length))

        try:
            features, mask = backbone(
                torch.randn(1, 3, 240, 240),
                torch.zeros(1, 240, 240, dtype=torch.bool),
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual([feature.shape for feature in features], [(1, 192, 15, 15)] * 4)
        self.assertEqual(mask.shape, (1, 15, 15))
        self.assertEqual(sorted(sequence_lengths[0]), [49, 56, 56, 64])
        self.assertEqual(sequence_lengths[1], [225])
        self.assertEqual(sorted(sequence_lengths[2]), [49, 56, 56, 64])
        self.assertEqual(sequence_lengths[3], [225])

    def test_dinov2_windowed_layer_blocks_cross_window_information(self) -> None:
        class TokenMixingLayer(nn.Module):
            def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor]:
                return (tokens + tokens.mean(dim=1, keepdim=True),)

        layer = DINOv2WindowedLayer(TokenMixingLayer(), num_windows=2)
        layer.set_spatial_shape(4, 4)
        tokens = torch.zeros(1, 17, 4)
        baseline = layer(tokens)[0]
        changed_tokens = tokens.clone()
        changed_tokens[:, -1] = 10.0
        changed = layer(changed_tokens)[0]

        # 最后一个 patch 位于右下窗口，不应改变左上窗口的第一个 patch。
        self.assertTrue(torch.equal(baseline[:, 1], changed[:, 1]))
        self.assertFalse(torch.equal(baseline[:, -1], changed[:, -1]))

    def test_dinov2_backbone_switches_attention_with_real_hf_blocks(self) -> None:
        from transformers import Dinov2Config, Dinov2Model

        dinov2 = Dinov2Model(
            Dinov2Config(
                image_size=32,
                patch_size=8,
                hidden_size=48,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=96,
            )
        )
        with patch("transformers.AutoModel.from_pretrained", return_value=dinov2):
            backbone = DINOv2Backbone(
                "offline-test",
                feature_layers=[1, 2, 3, 4],
                freeze=False,
                num_windows=2,
                window_block_indexes=[0, 2],
            )

        features, mask = backbone(
            torch.randn(2, 3, 40, 40),
            torch.zeros(2, 40, 40, dtype=torch.bool),
        )
        self.assertEqual([feature.shape for feature in features], [(2, 48, 5, 5)] * 4)
        self.assertEqual(mask.shape, (2, 5, 5))
        sum(feature.mean() for feature in features).backward()
        self.assertIsNotNone(dinov2.embeddings.patch_embeddings.projection.weight.grad)

    def test_forward_backward_contract(self) -> None:
        config = load_config(ROOT / "configs/local_smoke.yaml")
        model = build_model(config)
        images = torch.randn(2, 3, 224, 224)
        masks = torch.zeros(2, 224, 224, dtype=torch.bool)
        masks[:, 200:, :] = True
        targets = [
            {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([0])},
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
        ]
        outputs = model(images, masks)
        self.assertEqual(outputs["pred_logits"].shape, (2, 50, 1))
        self.assertEqual(outputs["pred_boxes"].shape, (2, 50, 4))
        self.assertTrue(bool(torch.isfinite(outputs["pred_boxes"]).all()))
        self.assertGreaterEqual(float(outputs["pred_boxes"].detach().min()), 0.0)
        self.assertLessEqual(float(outputs["pred_boxes"].detach().max()), 1.0)
        losses = SetCriterion(1)(outputs, targets)
        self.assertTrue(bool(torch.isfinite(losses["loss_total"])))
        losses["loss_total"].backward()

    def test_active_subnet_changes_shapes(self) -> None:
        config = load_config(ROOT / "configs/local_smoke.yaml")
        model = build_model(config)
        model.set_active_subnet(224, 25, 1)
        outputs = model(torch.randn(1, 3, 224, 224), torch.zeros(1, 224, 224, dtype=torch.bool))
        self.assertEqual(outputs["pred_logits"].shape, (1, 25, 1))
        self.assertEqual(outputs["aux_outputs"], [])

    def test_pareto_frontier(self) -> None:
        results = [
            {"name": "small", "map": 0.30, "latency_ms": 5.0},
            {"name": "dominated", "map": 0.25, "latency_ms": 8.0},
            {"name": "large", "map": 0.40, "latency_ms": 10.0},
        ]
        self.assertEqual([item["name"] for item in pareto_frontier(results)], ["small", "large"])


if __name__ == "__main__":
    unittest.main()
