import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_images = 0

    progress = tqdm(loader, desc="Train")

    for images, boxes, labels, cells in progress:
        images = images.to(device)

        # 1. 前向传播，计算Loss
        pred = model(images)
        loss = criterion(pred, boxes, labels, cells)

        if not torch.isfinite(loss):
            raise ValueError("训练Loss出现NaN或Inf，请检查学习率和预测值")

        # 2. 清空梯度，反向传播，更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. 按图片数量计算平均Loss
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_images += batch_size

        progress.set_postfix(loss=f"{total_loss / total_images:.4f}")

    if total_images == 0:
        raise ValueError("训练集没有图片")

    return total_loss / total_images
