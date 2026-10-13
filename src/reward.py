"""Module tính hàm reward IoU phục vụ RL/GRPO cho bài toán định vị hư hại mặt đường (E5).
Đảm bảo kiểm tra cú pháp JSON, tính IoU@0.5 giữa Ground Truth và Prediction, phạt box hallucination.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from scripts.labcommon import CLASSES
from scripts.score import iou_xyxy, parse_output

IOU_THRESHOLD = 0.5


def compute_iou_reward(
    completion: str,
    gt_boxes: List[Dict[str, Any]],
    iou_thr: float = IOU_THRESHOLD,
) -> float:
    """Tính reward cho một chuỗi output (completion) so với ground truth boxes.

    Công thức theo đề bài (GUIDE bước 9, configs/e5.yaml):
    - Nếu parse_output thất bại (JSON hỏng) -> 0.0
    - Nếu cả gt và pred đều rỗng -> 1.0
    - Ghép đôi tham lam theo IoU giảm dần, cùng nhãn lớp, chỉ nhận IoU >= 0.5
    - Reward = 0.2 + 0.8 * (tổng_iou_khớp) / max(n_gt, n_pred, 1)

    Args:
        completion: Chuỗi văn bản do mô hình sinh ra.
        gt_boxes: Danh sách các dict [{"label": str, "bbox": (x1, y1, x2, y2)}] thang 0-1000.
        iou_thr: Ngưỡng IoU tối thiểu để coi là khớp (mặc định 0.5).

    Returns:
        float: Giá trị phần thưởng trong khoảng [0.0, 1.0].
    """
    preds = parse_output(completion)
    if preds is None:
        return 0.0

    n_gt = len(gt_boxes)
    n_pred = len(preds)

    if n_gt == 0 and n_pred == 0:
        return 1.0

    # Ghép đôi tham lam cho từng lớp hoặc trên toàn bộ các cặp cùng lớp
    pairs = []
    for g_idx, g in enumerate(gt_boxes):
        for p_idx, p in enumerate(preds):
            if g["label"] == p["label"]:
                iou_val = iou_xyxy(g["bbox"], p["bbox"])
                if iou_val >= iou_thr:
                    pairs.append((iou_val, g_idx, p_idx))

    # Sắp xếp IoU giảm dần
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_gt = set()
    used_pred = set()
    total_iou = 0.0

    for iou_val, g_idx, p_idx in pairs:
        if g_idx in used_gt or p_idx in used_pred:
            continue
        used_gt.add(g_idx)
        used_pred.add(p_idx)
        total_iou += iou_val

    denom = max(n_gt, n_pred, 1)
    reward = 0.2 + 0.8 * (total_iou / denom)
    return float(reward)
