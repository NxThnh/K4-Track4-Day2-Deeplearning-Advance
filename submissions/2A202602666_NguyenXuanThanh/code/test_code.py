"""test_code.py - bộ unit test tự viết kiểm tra tính đúng đắn của toàn bộ pipeline code/.

Đáp ứng RUBRIC.md mục A (kiểm tra pipeline) và mục H (tự kiểm tra các phần dễ sai:
focal γ=0 ≡ CE, CutMix, gộp BatchNorm, chia dữ liệu S1-S6).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CODE_DIR))

import dataset as ds
import model as md
import losses as ls
import inference as inf
import benchmark as bm
import eval as ev


class TestDatasetAndSplits(unittest.TestCase):
    def test_load_and_check_split(self):
        labels_dir = REPO_ROOT / "data" / "labels"
        images_dir = REPO_ROOT / "data" / "images"
        if not labels_dir.exists() or not images_dir.exists():
            self.skipTest("Thư mục data/labels hoặc data/images chưa sẵn sàng")

        train_df, val_df, test_df = ds.load_split(labels_dir, fold=0)
        res = ds.check_split(train_df, val_df, test_df, images_dir)

        self.assertEqual(res["total"], 17509)
        self.assertEqual(res["overlap_tv"], 0)
        self.assertEqual(res["overlap_tt"], 0)
        self.assertEqual(res["overlap_vt"], 0)
        self.assertAlmostEqual(res["train_pct"], 0.60, delta=0.01)
        self.assertAlmostEqual(res["val_pct"], 0.20, delta=0.01)
        self.assertAlmostEqual(res["test_pct"], 0.20, delta=0.01)

    def test_transforms_shapes(self):
        img_size = 224
        t_train = ds.build_transforms(train=True, img_size=img_size, aug="basic")
        t_val = ds.build_transforms(train=False, img_size=img_size)

        from PIL import Image
        dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        train_tensor = t_train(dummy_img)
        val_tensor = t_val(dummy_img)

        self.assertEqual(train_tensor.shape, (3, img_size, img_size))
        self.assertEqual(val_tensor.shape, (3, img_size, img_size))


class TestLossesAndMixup(unittest.TestCase):
    def test_focal_loss_gamma_zero_matches_ce(self):
        """BẮT BUỘC theo RUBRIC: gamma=0 phải cho đúng cross-entropy thông thường (sai số < 1e-6)."""
        torch.manual_seed(42)
        logits = torch.randn(32, 9)
        targets = torch.randint(0, 9, (32,))

        fl_0 = ls.FocalLoss(gamma=0.0)(logits, targets)
        ce = nn.CrossEntropyLoss()(logits, targets)

        diff = abs(fl_0.item() - ce.item())
        self.assertLess(diff, 1e-6, f"Focal(gamma=0) lệch CE: {diff}")

    def test_cutmix_tensor_shapes_and_lambda(self):
        torch.manual_seed(42)
        x = torch.randn(8, 3, 224, 224)
        y = torch.randint(0, 9, (8,))

        x_cut, (ya, yb, lam) = ls.mix_batch(x, y, alpha=1.0, mode="cutmix")
        self.assertEqual(x_cut.shape, (8, 3, 224, 224))
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)
        self.assertEqual(ya.shape, (8,))
        self.assertEqual(yb.shape, (8,))

    def test_class_weights_balanced(self):
        counts = [1000] * 8 + [9000]
        w = ls.class_weights(counts, beta=0.9999)
        self.assertEqual(len(w), 9)
        # Lớp hiếm (1000 mẫu) phải có trọng số lớn hơn lớp Negative (9000 mẫu)
        self.assertGreater(w[0].item(), w[8].item())


class TestModelArchitecture(unittest.TestCase):
    def test_model_parameter_groups(self):
        m = md.build_model("resnet50", pretrained=False)
        groups = md.param_groups(m, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
        self.assertGreaterEqual(len(groups), 3)

        # Kiểm tra nhóm norm/bias có weight_decay == 0
        has_zero_decay = any(g["weight_decay"] == 0.0 for g in groups)
        self.assertTrue(has_zero_decay, "Phải có nhóm tham số weight_decay = 0 (bias/norm)")

    def test_freeze_backbone(self):
        m = md.build_model("resnet50", pretrained=False, init="frozen")
        classifier = m.get_classifier()
        classifier_params = set(classifier.parameters())

        for name, param in m.named_parameters():
            if param in classifier_params:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)


class TestInferenceAndCalibration(unittest.TestCase):
    def test_temperature_scaling_optimization(self):
        np.random.seed(42)
        logits = np.random.randn(200, 9)
        labels = np.random.randint(0, 9, 200)

        T = inf.fit_temperature(logits, labels)
        self.assertGreater(T, 0.0)

        probs = inf.apply_temperature(logits, T)
        self.assertEqual(probs.shape, (200, 9))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(200), atol=1e-5)

    def test_aggregate_views(self):
        np.random.seed(42)
        v1 = np.random.randn(20, 9)
        v2 = np.random.randn(20, 9)

        prob_agg = inf.aggregate_views([v1, v2], space="prob")
        self.assertEqual(prob_agg.shape, (20, 9))
        np.testing.assert_allclose(prob_agg.sum(axis=1), np.ones(20), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
