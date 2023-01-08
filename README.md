<center><img src="https://github.com/xuebinqin/U-2-Net/blob/master/figures/motor-demo.gif"></center>

# U-Net: 使用 PyTorch 进行语义分割

- [快速开始](#快速开始)
- [使用](#使用)
  - [训练](#训练)
  - [预测](#预测)
- [W&B可视化](#W&B可视化)
- [预训练权重](#预训练权重)
- [数据集](#数据集)

## 快速开始

1. [安装 CUDA](https://developer.nvidia.com/cuda-downloads)

2. [安装 PyTorch 1.12 或更新的版本](https://pytorch.org/get-started/locally/)

3. 安装依赖
```bash
pip install -r requirements.txt
```

```
torch                          1.13.0
torchvision                    0.14.0
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

## 使用

**注意 : 请使用 Python 3.6 或更新的版本**

### 训练

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

默认情况下，`scale`为 0.5，因此如果您希望获得更好的结果（会被使用更多内存），请将其设置为 1。

`--amp` 代表使用自动混合精度。混合精度允许模型使用更少的内存，并通过使用 FP16 算法在 GPU 上更快，建议启用 AMP。

- 支持训练的格式搭配：

|imgs|masks|
|:-|:-|
|jpg|gif|
|tif|gif|

- 实验结果：

|model|Validation Dice score|
|:-|:-|
|UNet|0.99|


### 预测

训练模型并将其保存到`MODEL.pth`，您可以通过 CLI（命令行） 轻松测试图像上的输出掩码。

要预测单个图像并保存它：

`python predict.py -i image.jpg -o output.jpg`

要预测多个图像并在不保存的情况下显示它们：

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```

您可以使用 `--model MODEL.pth` 指定要使用的模型文件。

## W&B可视化

可以使用[Weights & Biases](https://wandb.ai/) 实时可视化训练进度。损失曲线、验证曲线、权重和梯度直方图以及预测掩码都记录到平台上。
 
启动train时，控制台中会打印一个链接。单击它转到您的仪表板。如果您已有 W&B 帐户，则可以通过设置 `WANDB_API_KEY` 环境变量来链接它。如果没有，它将创建一个匿名运行，并在 7 天后自动删除。


## 预训练权重

在Carvana 数据集上训练的[预训练模型](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) 。也可以从 torch.hub 加载：

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
```

可用scales为 0.5 和 1.0。

## 数据集

Carvana 数据可在 [Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge/data) 上获得。

您也可以使用帮助脚本下载它：

```
bash scripts/download_data.sh
```

输入图像和目标蒙版应分别位于 `data/imgs` 和 `data/masks` 文件夹中（请注意，由于数据加载器，`imgs` 和 `masks` 文件夹不应包含任何子文件夹或任何其他文件）。对于 Carvana，图像是 RGB，蒙版是黑白的。

```
- imgs：.jpg
- masks：.gif
```

只要确保在 `utils/data_loading.py`中正确加载它，就可以使用自己的数据集。


## 参考

- [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- [https://www.cnblogs.com/wanghui-garcia/p/10719121.html](https://www.cnblogs.com/wanghui-garcia/p/10719121.html)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)


<center><img src="https://i.imgur.com/jeDVpqF.png" alt="network architecture"></center>
