import torch
from tqdm import tqdm
from torchvision.ops import box_convert
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from Utils.metric import decode_predictions


@torch.no_grad()
def val(model, loader, criterion, device):
    model.eval()

    # 使用现成的COCO风格mAP，不自己手写AP积分
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    total_loss = 0.0
    total_images = 0

    progress = tqdm(loader, desc="Val")

    for images, boxes, labels, cells in progress:
        images = images.to(device)

        # 1. 计算验证Loss
        pred = model(images)
        loss = criterion(pred, boxes, labels, cells)

        if not torch.isfinite(loss):
            raise ValueError("验证Loss出现NaN或Inf")
        if not torch.isfinite(pred).all():
            raise ValueError("验证预测出现NaN或Inf")

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size

        # 2. 解码预测框，转为mAP需要的boxes、scores、labels
        H, W = images.shape[-2:]
        predictions = decode_predictions(pred, (H, W))

        # 3. GT也转为同一张图片上的像素xyxy
        targets = []
        for i in range(batch_size):
            gt_boxes = box_convert(boxes[i].cpu(), "cxcywh", "xyxy")
            gt_boxes = gt_boxes * gt_boxes.new_tensor([W, H, W, H])

            # 评估保留全部GT，包括同一Cell中的多个目标
            targets.append({"boxes": gt_boxes, "labels": labels[i].cpu()})

        # 4. 累积整个验证集，最后只计算一次mAP
        metric.update(predictions, targets)
        progress.set_postfix(loss=f"{total_loss / total_images:.4f}")

    if total_images == 0:
        raise ValueError("验证集没有图片")

    result = metric.compute()

    return {
        "loss": total_loss / total_images,
        "map50": result["map_50"].item(),
        "map50_95": result["map"].item(),
    }
