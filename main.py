import json
import os

import torch

from Dataset.dataloader import get_loader
from Model.YoloV1 import YoloV1
from Utils.loss import YoloV1Loss
from Train.train_one_epoch import train_one_epoch
from Train.val import val
from predict_val import predict_val

# 1. 固定参数，想改什么就直接改这里
# 在YoloV1项目目录里运行：python main.py
# 服务器datasets目录里面应该包含VOCdevkit

data_root = "/home/gpu/datasets"
save_dir = "./checkpoints"
prediction_dir = "./val_predictions"

epochs = 50
batch_size = 6
lr = 0.0001
num_workers = 0

# None表示使用全部数据；本机调试可以改成16和10
train_samples = None
val_samples = None


def main():
    if epochs <= 0:
        raise ValueError("epochs必须大于0")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 2. DataLoader
    train_loader = get_loader(
        "train",
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=train_samples,
    )
    val_loader = get_loader(
        "val",
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=val_samples,
    )
    print("train images:", len(train_loader.dataset))
    print("val images:", len(val_loader.dataset))

    # 3. 模型、Loss、优化器
    model = YoloV1().to(device)
    criterion = YoloV1Loss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )

    # 单独放一个模型加载最好权重，继续训练时仍然使用上面的model
    best_model = YoloV1().to(device)
    os.makedirs(save_dir, exist_ok=True)
    best_map50 = -1.0
    best_epoch = 0
    history = []

    # 4. 每轮先训练，再验证
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = val(model, val_loader, criterion, device)

        print(
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_result['loss']:.4f}  "
            f"mAP@0.5={val_result['map50']:.4f}  "
            f"mAP@0.5:0.95={val_result['map50_95']:.4f}"
        )

        # 5. 用验证集mAP@0.5选择最好模型
        if val_result["map50"] > best_map50:
            best_map50 = val_result["map50"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"保存最好模型，mAP@0.5={best_map50:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # 6. 每轮结束，加载目前最好的权重，预测val前10张
        best_weights = torch.load(
            os.path.join(save_dir, "best_model.pth"),
            map_location=device,
            weights_only=True,
        )
        best_model.load_state_dict(best_weights)
        epoch_dir = os.path.join(prediction_dir, f"epoch_{epoch + 1:03d}")
        predict_val(
            best_model,
            data_root,
            epoch_dir,
            device,
            num_images=10,
            best_epoch=best_epoch,
            best_map50=best_map50,
        )

        # 7. 保存每轮数字，方便看Loss和mAP的变化
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_result["loss"],
                "map50": val_result["map50"],
                "map50_95": val_result["map50_95"],
                "best_epoch": best_epoch,
            }
        )
        with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"\n训练结束，最好mAP@0.5={best_map50:.4f}，来自第{best_epoch}轮")
    return history


if __name__ == "__main__":
    main()
