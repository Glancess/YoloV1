# YOLOv1 训练

在YoloV1项目目录里运行：

```bash
python -m pip install -r requirements.txt
python main.py
```

所有参数直接写在main.py开头，不使用命令行参数：

```python
data_root = "./Data"
save_dir = "./checkpoints"
prediction_dir = "./val_predictions"

epochs = 50
batch_size = 2
lr = 0.0001
num_workers = 0
train_samples = None
val_samples = None
```

数据目录应为 `Data/VOCdevkit/VOC2012`。数据在别的位置就直接修改data_root。
None表示完整数据集；本机检查流程时，可以直接改成epochs=3、train_samples=16、val_samples=10。
服务器正式训练时把train_samples和val_samples改回None。
ResNet18使用ImageNet预训练权重，首次需要下载或提前缓存。

代码结构：

```text
Dataset/dataset.py          读取VOC，返回image、boxes、labels、cells
Dataset/dataloader.py       图片组成Batch，标注使用列表
Model/YoloV1.py             ResNet18 + 检测头
Utils/loss.py               YOLOv1 Loss
Utils/metric.py             解码 + torchvision按类别NMS
Train/train_one_epoch.py    训练一轮
Train/val.py                验证Loss和mAP
predict_val.py             画val前10张图片的GT和预测框
main.py                    每轮训练、验证、保存权重和预测图
```

每轮结束会：

1. 在验证集上计算loss、mAP@0.5和mAP@0.5:0.95。
2. 按mAP@0.5保存目前最好的best_model.pth，另外保存本轮last_model.pth。
3. 用单独的best_model加载最好权重，预测固定的val前10张。
4. 将10张对比图、总览和分数保存到本轮文件夹。

例如第2轮mAP低于第1轮，第2轮输出图仍然使用第1轮的最好权重。
预测使用单独模型，训练的model和优化器会继续当前训练进度。

```text
checkpoints/
  best_model.pth
  last_model.pth
  history.json
val_predictions/
  epoch_001/
    01_2008_000002.png
    ...共10张PNG
    overview.jpg
    predictions.json
  epoch_002/
  ...
```

每张图左边绿色为GT，右边红色为预测框和类别分数。显示阈值为0.10，
每图最多显示10个预测框；没有达到阈值的预测时会注明。
图标题和predictions.json记录使用的最好权重来自第几轮。

训练后也可单独运行 `python predict_val.py`，读取checkpoints/best_model.pth，
输出到val_predictions/manual。这些pth文件只保存model.state_dict()，不含优化器状态。
再次训练使用相同目录时会覆盖同名文件。

mAP使用torchmetrics的MeanAveragePrecision和pycocotools，分数范围0到1。
mAP@0.5用于选择最好权重，mAP@0.5:0.95更严格。
指标累积整个验证集后计算一次，mAP的score阈值为0.001、NMS阈值为0.5，
每张图片最多100个预测，与显示阈值分开。
当前所有VOC标注包括difficult都作为普通GT，属于COCO风格mAP，
不等同于VOC官方评测口径。同Cell多个GT在评估时全部保留，Loss沿用原实现只监督第一个。
少量图片跑通只能验证流程，模型效果需要完整验证集mAP判断。
