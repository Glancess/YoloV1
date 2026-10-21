import torch
import torch.nn as nn
from torchvision.ops import box_convert, box_iou


class YoloV1Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super().__init__()

        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, pred, boxes, labels, cells):

        batch_size = pred.shape[0]

        # pred: [batch_size, 7, 7, 30]
        box_preds = pred[..., :10].reshape(batch_size, 7, 7, 2, 5)

        # 每个Cell只有一套类别预测
        cls_preds = pred[..., 10:]  # [batch_size,7,7,20]

        # 初始化五个Loss
        loss_xy = pred.new_tensor(0.0)
        loss_wh = pred.new_tensor(0.0)
        loss_obj = pred.new_tensor(0.0)
        loss_noobj = pred.new_tensor(0.0)
        loss_cls = pred.new_tensor(0.0)

        # 所有Box默认都不负责目标
        # 后面负责GT的Box要改为False
        noobj_mask = torch.ones(
            batch_size, 7, 7, 2, dtype=torch.bool, device=pred.device
        )

        # 1. 遍历Batch
        for i in range(batch_size):

            # 一个Cell只能监督一个GT
            used_cells = set()

            # 2. 遍历当前图片的GT
            for j in range(len(boxes[i])):

                gtbox = boxes[i][j].to(pred.device)
                row, col = cells[i][j].tolist()

                # 同一Cell有多个GT时，暂时只取第一个
                if (row, col) in used_cells:
                    continue

                used_cells.add((row, col))

                # 3. 找到该Cell的两个预测框
                cell_boxes = box_preds[i, row, col]  # [2,5]

                box1 = cell_boxes[0]
                box2 = cell_boxes[1]

                # 4. 计算IoU，选择负责的Box
                with torch.no_grad():

                    # 不要修改原始预测
                    pred_boxes = cell_boxes[:, :4].clone()

                    # Cell内xy -> 整图归一化cxcy
                    pred_boxes[:, 0] = (pred_boxes[:, 0] + col) / self.S

                    pred_boxes[:, 1] = (pred_boxes[:, 1] + row) / self.S

                    # 预测的wh可能为负数
                    pred_boxes[:, 2:4] = pred_boxes[:, 2:4].clamp(min=1e-6)

                    # cxcywh -> xyxy
                    pred_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")

                    gt_xyxy = box_convert(gtbox.unsqueeze(0), "cxcywh", "xyxy")

                    # [2,1] -> [2]
                    ious = box_iou(pred_xyxy, gt_xyxy).squeeze(1)

                    # IoU最大的Box负责
                    responsible_idx = ious.argmax().item()

                # 负责框不再属于noobj
                noobj_mask[i, row, col, responsible_idx] = False

                # 用原始预测计算Loss，保留梯度
                responsible_box = cell_boxes[responsible_idx]

                # -----------------------
                # Loss 1: xy
                # -----------------------

                # GT整图xy -> Cell内部xy
                gt_xy = gtbox[:2] * self.S
                gt_xy = gt_xy - gtbox.new_tensor([col, row])

                pred_xy = responsible_box[:2]

                loss_xy = loss_xy + ((pred_xy - gt_xy) ** 2).sum()

                # -----------------------
                # Loss 2: wh
                # -----------------------

                pred_wh = responsible_box[2:4]
                gt_wh = gtbox[2:4]

                # 最后一层是Linear，可能预测负宽高
                # 使用保留符号的sqrt避免NaN
                pred_wh_sqrt = torch.sign(pred_wh) * torch.sqrt(
                    torch.abs(pred_wh) + 1e-6
                )

                gt_wh_sqrt = torch.sqrt(gt_wh.clamp(min=1e-6))

                loss_wh = loss_wh + ((pred_wh_sqrt - gt_wh_sqrt) ** 2).sum()

                # -----------------------
                # Loss 3: obj confidence
                # -----------------------

                pred_conf = responsible_box[4]

                # 负责框的Confidence目标是IoU
                target_conf = ious[responsible_idx]

                loss_obj = loss_obj + (pred_conf - target_conf) ** 2

                # -----------------------
                # Loss 4: classification
                # -----------------------

                pred_cls = cls_preds[i, row, col]

                # 构造one-hot类别
                gt_cls = torch.zeros(self.C, device=pred.device, dtype=pred.dtype)

                class_id = int(labels[i][j])
                gt_cls[class_id] = 1.0

                loss_cls = loss_cls + ((pred_cls - gt_cls) ** 2).sum()

        # -----------------------
        # Loss 5: noobj confidence
        # -----------------------

        # 包括空Cell中的两个框，
        # 以及有GT的Cell中不负责的那个框
        all_conf = box_preds[..., 4]  # [batch_size,7,7,2]

        noobj_conf = all_conf[noobj_mask]

        loss_noobj = (noobj_conf**2).sum()

        # -----------------------
        # 汇总五项Loss
        # -----------------------

        total_loss = (
            self.lambda_coord * loss_xy
            + self.lambda_coord * loss_wh
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        )

        # 按Batch Size取平均，方便训练
        total_loss = total_loss / batch_size

        return total_loss


if __name__ == "__main__":
    from Model.YoloV1 import YoloV1
    from Dataset.dataloader import get_loader

    model = YoloV1()
    criterion = YoloV1Loss()
    loader = get_loader("train")
    images, boxes, labels, cells = next(iter(loader))

    pred = model(images)

    loss = criterion(pred, boxes, labels, cells)

    print("pred:", pred.shape)
    print("loss:", loss.item())

    loss.backward()

    print("Backward successful!")
