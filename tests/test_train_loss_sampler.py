import torch

from src.train import build_loss, build_sampler_weights


def _cfg(loss="dice_focal_bce"):
    return {
        "train": {
            "loss": loss,
            "loss_lambda_dice": 1.0,
            "loss_lambda_bce": 1.0,
            "loss_lambda_focal": 0.25,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "loss_lambda_skeleton": 0.2,
            "skeleton_target_threshold": 0.95,
            "skeleton_pos_weight": 8.0,
            "weighted_sampler": True,
            "hard_cases": ["052-31/32", "041-41"],
            "hard_case_weight": 4.0,
        }
    }


def test_new_losses_forward_backward():
    for loss_name in ("dice_focal_bce", "dice_focal_hard", "dice_focal_skeleton"):
        logits = torch.randn((2, 1, 6, 6, 6), requires_grad=True)
        target = torch.zeros((2, 1, 6, 6, 6))
        target[:, :, 2:4, 2:4, 2:4] = 1.0
        loss_fn = build_loss(_cfg(loss_name))
        loss = loss_fn(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_sampler_weights_match_dataset_length_and_hard_cases():
    class DummyDataset:
        items = [
            {"case_id": "ToothFairy3F_052", "tooth_id": 31},
            {"case_id": "ToothFairy3F_052", "tooth_id": 33},
            {"case_id": "ToothFairy3F_041", "tooth_id": 41},
            {"case_id": "ToothFairy3F_050", "tooth_id": 11},
        ]

    weights = build_sampler_weights(DummyDataset(), _cfg())
    assert len(weights) == len(DummyDataset.items)
    assert weights.tolist() == [4.0, 1.0, 4.0, 1.0]
