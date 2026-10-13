"""Unit tests cho hàm reward IoU trong GRPO (E5).
Kiểm tra các trường hợp: hoàn hảo, JSON hỏng, sai lớp, thừa box, cả hai rỗng, IoU dưới ngưỡng.
"""
import json
import pytest

from src.reward import compute_iou_reward


def test_reward_perfect_match():
    gt = [{"label": "pothole", "bbox": (100.0, 100.0, 200.0, 200.0)}]
    pred_str = json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}])
    reward = compute_iou_reward(pred_str, gt)
    # IoU = 1.0, denom = 1 -> reward = 0.2 + 0.8 * 1.0 = 1.0
    assert reward == pytest.approx(1.0)


def test_reward_both_empty():
    gt = []
    pred_str = "[]"
    assert compute_iou_reward(pred_str, gt) == 1.0


def test_reward_malformed_json():
    gt = [{"label": "pothole", "bbox": (100.0, 100.0, 200.0, 200.0)}]
    assert compute_iou_reward("not a valid json", gt) == 0.0
    assert compute_iou_reward("{label: pothole}", gt) == 0.0
    assert compute_iou_reward("", gt) == 0.0


def test_reward_wrong_class():
    gt = [{"label": "pothole", "bbox": (100.0, 100.0, 200.0, 200.0)}]
    pred_str = json.dumps([{"label": "crack", "bbox_2d": [100, 100, 200, 200]}])
    # Không khớp lớp -> total_iou = 0 -> reward = 0.2 + 0 = 0.2
    assert compute_iou_reward(pred_str, gt) == pytest.approx(0.2)


def test_reward_hallucinated_boxes_penalized():
    # Model hallucinate thêm nhiều box -> denom tăng lên max(n_gt, n_pred)
    gt = [{"label": "pothole", "bbox": (100.0, 100.0, 200.0, 200.0)}]
    pred_str = json.dumps([
        {"label": "pothole", "bbox_2d": [100, 100, 200, 200]},
        {"label": "pothole", "bbox_2d": [300, 300, 400, 400]},
        {"label": "crack", "bbox_2d": [500, 500, 600, 600]},
        {"label": "manhole", "bbox_2d": [700, 700, 800, 800]},
    ])
    reward = compute_iou_reward(pred_str, gt)
    # 1 cặp khớp IoU 1.0, nhưng n_pred = 4 -> denom = 4 -> total_iou/4 = 0.25 -> reward = 0.2 + 0.8 * 0.25 = 0.4
    assert reward == pytest.approx(0.4)


def test_reward_iou_below_threshold():
    gt = [{"label": "pothole", "bbox": (0.0, 0.0, 100.0, 100.0)}]
    # Box lệch nhiều, IoU < 0.5
    pred_str = json.dumps([{"label": "pothole", "bbox_2d": [50, 50, 150, 150]}])
    # Intersection = 50*50 = 2500, Union = 10000 + 10000 - 2500 = 17500 -> IoU = 2500/17500 = 0.1428 < 0.5
    assert compute_iou_reward(pred_str, gt) == pytest.approx(0.2)
