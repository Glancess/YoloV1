import torch
from torchvision.ops import batched_nms


def decode_predictions(pred, image_size, score_threshold=0.001, nms_threshold=0.5):
    # mAP和NMS在CPU计算，不需要梯度
    pred = pred.detach().cpu()
    H, W = image_size
    S = pred.shape[1]
    B = 2  # 和当前模型、Loss一致

    results = []

    # 1. 遍历Batch中的每张图片
    for image_pred in pred:
        boxes = []
        scores = []
        labels = []

        for row in range(S):
            for col in range(S):
                cell = image_pred[row, col]

                # 当前Loss用one-hot做回归，这里不额外加softmax
                class_scores = cell[B * 5:].clamp(0, 1).tolist()

                for b in range(B):
                    cx, cy, w, h, confidence = cell[b * 5:(b + 1) * 5].tolist()

                    # 负宽高的预测框没有意义，直接跳过
                    if w <= 0 or h <= 0:
                        continue

                    # Cell内xy -> 整张图片上的像素坐标
                    cx = (cx + col) / S * W
                    cy = (cy + row) / S * H
                    w = w * W
                    h = h * H

                    xmin = max(0.0, min(W, cx - w / 2))
                    ymin = max(0.0, min(H, cy - h / 2))
                    xmax = max(0.0, min(W, cx + w / 2))
                    ymax = max(0.0, min(H, cy + h / 2))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    confidence = max(0.0, min(1.0, confidence))

                    # 每个类别的分数 = 框Confidence × 类别分数
                    for class_id, class_score in enumerate(class_scores):
                        score = confidence * class_score

                        if score > score_threshold:
                            boxes.append([xmin, ymin, xmax, ymax])
                            scores.append(score)
                            labels.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        scores = torch.tensor(scores, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 2. 按类别做NMS，不同类别不会互相删除
        keep = batched_nms(boxes, scores, labels, nms_threshold)
        keep = keep[:100]  # 每张图片最多保留100个预测

        results.append(
            {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
        )

    return results
