import hashlib
import os
import subprocess

from torchvision.datasets import VOCDetection
from torchvision.datasets.utils import extract_archive


# 固定下载到服务器的datasets目录
# 解压后为 /home/gpu/datasets/VOCdevkit/VOC2012

data_root = "/home/gpu/datasets"
url = "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
filename = "VOCtrainval_11-May-2012.tar"
expected_size = 1999639040
expected_md5 = "6cd6e144f989b92b3379bac3b3de84fd"


def dataset_ready():
    voc_dir = os.path.join(data_root, "VOCdevkit", "VOC2012")
    for split in ["train", "val"]:
        split_file = os.path.join(voc_dir, "ImageSets", "Main", split + ".txt")
        if not os.path.isfile(split_file):
            return False
        with open(split_file) as f:
            image_ids = f.read().split()
        if not image_ids:
            return False
        for image_id in image_ids:
            image = os.path.join(voc_dir, "JPEGImages", image_id + ".jpg")
            annotation = os.path.join(voc_dir, "Annotations", image_id + ".xml")
            if not os.path.isfile(image) or not os.path.isfile(annotation):
                return False
    return True


def main():
    os.makedirs(data_root, exist_ok=True)

    if not dataset_ready():
        archive = os.path.join(data_root, filename)
        size = os.path.getsize(archive) if os.path.isfile(archive) else 0
        if size > expected_size:
            raise ValueError("已有压缩包大小不对，请先检查文件；本程序不会删除它")

        # -C - 表示断点续传，不丢掉之前下载的部分
        if size < expected_size:
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "3",
                 "--connect-timeout", "20", "--continue-at", "-",
                 "--output", archive, url],
                check=True,
            )

        # 完整下载后校验，确认文件没有损坏再解压
        print("正在校验VOC2012压缩包...", flush=True)
        md5 = hashlib.md5()
        with open(archive, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                md5.update(chunk)
        if md5.hexdigest() != expected_md5:
            raise ValueError("VOC2012校验失败，请检查下载文件；本程序不会删除它")

        print("校验通过，正在解压...", flush=True)
        extract_archive(archive, data_root)
        if not dataset_ready():
            raise ValueError("解压后仍有图片或标注缺失，请检查datasets目录")

    train = VOCDetection(root=data_root, year="2012", image_set="train", download=False)
    val = VOCDetection(root=data_root, year="2012", image_set="val", download=False)
    print("数据目录:", data_root)
    print("train:", len(train), "val:", len(val))
    print("数据已就绪，可以运行 python main.py")


if __name__ == "__main__":
    main()
