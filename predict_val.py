import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
from torchvision.ops import box_convert

from Dataset.dataset import MyDataset, VOC_CLASSES
from Model.YoloV1 import YoloV1
from Utils.metric import decode_predictions


def draw_boxes(ax, boxes, labels, color, scores=None):
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i].tolist()
        ax.add_patch(
            Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                edgecolor=color, facecolor="none", linewidth=2,
            )
        )

        text = VOC_CLASSES[int(labels[i])]
        if scores is not None:
            text += f" {scores[i].item():.3f}"

        ax.text(
            xmin, max(12, ymin), text, color="white", fontsize=9,
            bbox={"facecolor": color, "alpha": 0.7, "pad": 2},
        )


@torch.no_grad()
def predict_val(model, data_root, save_dir, device, num_images=10,
                best_epoch=None, best_map50=None):
    # 训练时main会把最好的模型传进来；这里不训练，只预测和画图
    score_threshold = 0.10
    max_boxes = 10
    dataset = MyDataset(data_root, "val")
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    records = []
    image_paths = []

    # 固定取val前10张，不挑选预测效果好的图片
    for i in range(min(num_images, len(dataset))):
        image, boxes, labels, cells = dataset[i]
        H, W = image.shape[-2:]
        image_id = os.path.splitext(os.path.basename(dataset.voc.images[i]))[0]

        # 1. 预测，使用验证函数里同一套解码和NMS
        pred = model(image.unsqueeze(0).to(device))
        result = decode_predictions(pred, (H, W))[0]
        max_score = result["scores"].max().item() if len(result["scores"]) else 0.0

        # 显示用0.10阈值；mAP计算仍保留原来的0.001阈值
        keep = torch.where(result["scores"] >= score_threshold)[0][:max_boxes]
        pred_boxes = result["boxes"][keep]
        pred_labels = result["labels"][keep]
        pred_scores = result["scores"][keep]

        # 2. 还原GT框，也还原图片的Normalize，避免显示颜色不对
        gt_boxes = box_convert(boxes, "cxcywh", "xyxy")
        gt_boxes = gt_boxes * torch.tensor([W, H, W, H])
        display_image = (image * std + mean).clamp(0, 1)
        display_image = display_image.permute(1, 2, 0).numpy()

        # 3. 左边GT，右边预测
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax in axes:
            ax.imshow(display_image)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")

        draw_boxes(axes[0], gt_boxes, labels, color="green")
        draw_boxes(axes[1], pred_boxes, pred_labels, color="red", scores=pred_scores)
        axes[0].set_title(f"GT: {len(gt_boxes)} objects")
        axes[1].set_title(
            f"Prediction: score >= {score_threshold:.2f}, {len(pred_boxes)} boxes"
        )
        if len(pred_boxes) == 0:
            axes[1].text(
                10, 25, f"No predictions >= {score_threshold:.2f}\nMax score: {max_score:.3f}",
                color="white", fontsize=11, bbox={"facecolor": "black", "alpha": 0.7},
            )

        title = f"{i + 1:02d} | {image_id} | best checkpoint"
        if best_epoch is not None:
            title += f" from epoch {best_epoch}"
        fig.suptitle(title)
        plt.tight_layout()
        image_path = os.path.join(save_dir, f"{i + 1:02d}_{image_id}.png")
        fig.savefig(image_path, dpi=120)
        plt.close(fig)
        image_paths.append(image_path)

        records.append(
            {
                "image_id": image_id,
                "max_score": max_score,
                "shown_boxes": len(pred_boxes),
                "gt_boxes_xyxy": gt_boxes.tolist(),
                "gt_classes": [VOC_CLASSES[int(label)] for label in labels],
                "pred_boxes_xyxy": pred_boxes.tolist(),
                "pred_classes": [VOC_CLASSES[int(label)] for label in pred_labels],
                "pred_scores": pred_scores.tolist(),
            }
        )
        print(f"{image_id}: GT={len(gt_boxes)}, shown={len(pred_boxes)}, max_score={max_score:.3f}")

    # 4. 把10张对比图拼成总览，单张原图仍然单独保存
    rows = (len(image_paths) + 1) // 2
    overview = Image.new("RGB", (1440, rows * 360), "white")
    for i, path in enumerate(image_paths):
        with Image.open(path) as im:
            overview.paste(im.resize((720, 360)), ((i % 2) * 720, (i // 2) * 360))
    overview.save(os.path.join(save_dir, "overview.jpg"), quality=95)

    with open(os.path.join(save_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"best_epoch": best_epoch, "best_map50": best_map50,
             "score_threshold": score_threshold, "images": records},
            f, indent=2,
        )
    print("结果保存到:", save_dir)
    return records


if __name__ == "__main__":
    # 单独看结果时，在项目目录运行：python predict_val.py
    data_root = "./Data"
    model_path = "./checkpoints/best_model.pth"
    save_dir = "./val_predictions/manual"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YoloV1().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    predict_val(model, data_root, save_dir, device, num_images=10)
