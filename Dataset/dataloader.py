import torch
from torch.utils.data import DataLoader, Subset

from .dataset import MyDataset


def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    cells = []

    for image, box, label, cell in batch:
        images.append(image)
        boxes.append(box)
        labels.append(label)
        cells.append(cell)
    images = torch.stack(images, dim=0)
    return images, boxes, labels, cells


def get_loader(which, root="./Data", batch_size=2, num_workers=0, max_samples=None):
    # 1. 创建 Dataset
    dataset = MyDataset(
        root=root, which=which
    )
    # 少量图片调试时使用，默认使用完整数据集
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples必须大于0")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    # 2. 创建 DataLoader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(which == "train"),
        num_workers=num_workers,  # 默认用0，服务器上可调大
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return loader


# 3. 测试一个 Batch
if __name__ == "__main__":
    loader = get_loader("train")
    images, boxes, labels, cells = next(iter(loader))

    print("images:", images.shape)

    for i in range(len(boxes)):
        print(f"Image {i}:")
        print("  boxes:", boxes[i].shape)
        print("  labels:", labels[i].shape)
        print("  cells:", cells[i].shape)
