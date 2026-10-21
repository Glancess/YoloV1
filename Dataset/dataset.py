from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
import torch
from torchvision.ops import box_convert

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root, which):
        super().__init__()

        self.voc = VOCDetection(root=root, year="2012", image_set=which, download=False)

        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, annotation = self.voc[idx]
        W, H = image.size
        image = self.transform(image)
        objects = annotation["annotation"]["object"]

        boxes = []
        labels = []

        for obj in objects:
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASSES.index(obj["name"]))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
        # Yolo V1 需要 cx cy，以及wh（中心+wh）
        cells = []

        for box in boxes:
            # W、H 是 Resize 前的原图尺寸
            cx, cy, w, h = box

            S = 7

            col = min(int(cx / W * S), S - 1)
            row = min(int(cy / H * S), S - 1)
            cells.append([row, col])

        boxes = boxes / torch.tensor([W, H, W, H], dtype=torch.float32)  # 归一化
        cells = torch.tensor(cells, dtype=torch.long)
        return image, boxes, labels, cells


if __name__ == "__main__":
    dataset = MyDataset("/home/gpu/datasets", "train")

    image, boxes, labels, cells = dataset[0]
    print(boxes)
    print(cells)
