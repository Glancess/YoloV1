import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class YoloV1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()

        self.S = S
        self.B = B
        self.C = C

        # ImageNet 预训练 ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 删除原本的 AvgPool 和 FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # [B,512,14,14] -> [B,256,7,7]
        self.neck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # [B,256,7,7] -> [B,1470]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x.reshape(x.shape[0], self.S, self.S, self.B * 5 + self.C)


if __name__ == "__main__":
    model = YoloV1()

    x = torch.randn(1, 3, 448, 448)

    model.eval()
    with torch.inference_mode():
        output = model(x)

    print(output.shape)
    # torch.Size([1, 7, 7, 30])
