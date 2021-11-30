## [YOLO v5](https://github.com/ultralytics/yolov5)自定义数据集检测

### YOLO v5训练自己数据集详细教程


#### 0、环境配置

安装必要的python package和配置相关环境

```
# python3.6
# torch==1.3.0
# torchvision==0.4.1

# 安装必要的package
pip3 install -U -r requirements.txt
```

#### 1、创建数据集的配置文件`dataset.yaml`

[data/coco128.yaml](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)来自于COCO train2017数据集的前128个训练图像，可以基于该`yaml`修改自己数据集的`yaml`文件

 ```ymal
# train and val datasets (image directory or *.txt file with image paths)
train: ./datasets/balloon/images/train/
val: ./datasets/balloon/images/val/

# number of classes
nc: 1

# class names
names: ['balloon']
 ```

#### 2、创建标注文件

可以使用LabelImg、Labme、[Labelbox](https://labelbox.com/)、[CVAT](https://github.com/opencv/cvat)来标注数据，对于目标检测而言需要标注bounding box即可。然后需要将标注转换为和**darknet format**相同的标注形式，每一个图像生成一个`*.txt`的标注文件（如果该图像没有标注目标则不用创建`*.txt`文件）。创建的`*.txt`文件遵循如下规则：

- 每一行存放一个标注类别
- 每一行的内容包括`class x_center y_center width height`
- Bounding box 的坐标信息是归一化之后的（0-1）
- class label转化为index时计数是从0开始的

```python
def convert(size, box):
    '''
    将标注的xml文件标注转换为darknet形的坐标
    '''
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
```

每一个标注`*.txt`文件存放在和图像相似的文件目录下，只需要将`/images/*.jpg`替换为`/lables/*.txt`即可（这个在加载数据时代码内部的处理就是这样的，可以自行修改为VOC的数据格式进行加载）

例如：

```
datasets/score/images/train/000000109622.jpg  # image
datasets/score/labels/train/000000109622.txt  # label
```
如果一个标注文件包含5个person类别（person在coco数据集中是排在第一的类别因此index为0）：

标注文件的内容如下（坐标使用归一化之后的）：

``` python
# 类别id centerx centery width height
0 0.654785 0.559245 0.135742 0.160156
0 0.668457 0.685547 0.168945 0.144531
```

#### 3、组织训练集的目录

将训练集train和验证集val的images和labels文件夹按照如下的方式进行存放

``` python
./datasets/
|——balloon/
	|——images/
		|——train/
			|——001.jpg
			|——002.jpg
		|——val/
			|——101.jpg
			|——102.jpg
		|——val.shape文件
	|——labels/
		|——train/
			|——001.txt
			|——002.txt
		|——val/
			|——101.txt
			|——102.txt
```

至此数据准备阶段已经完成，过程中我们假设算法工程师的数据清洗和数据集的划分过程已经自行完成。

#### 4、选择模型backbone进行模型配置文件的修改

在项目的`./models`文件夹下选择一个需要训练的模型，这里我们选择[yolov5s.yaml](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml)，最大的一个模型进行训练，参考官方README中的[table](https://github.com/ultralytics/yolov5#pretrained-checkpoints)，了解不同模型的大小和推断速度。如果你选定了一个模型，那么需要修改模型对应的`yaml`文件

```yaml
# parameters
nc: 1  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# yolov5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 1-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 3, Bottleneck, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 6-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 8-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 6, BottleneckCSP, [1024]],  # 10
  ]

# yolov5 head
head:
  [[-1, 3, BottleneckCSP, [1024, False]],  # 11
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 12 (P5/32-large)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 17 (P4/16-medium)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 22 (P3/8-small)

   [[], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 5、Train

```bash
# Train yolov5s on score for 300 epochs

$ python3 train.py --img-size 640 --batch-size 16 --epochs 300 --data ./data/balloon.yaml --cfg ./models/balloon/yolov5s.yaml --weights weights/yolov5s.pt
```


#### 6、Visualize

开始训练后，查看`train*.jpg`图片查看训练数据，标签和数据增强，如果你的图像显示标签或数据增强不正确，你应该查看你的数据集的构建过程是否有问题。

一个训练epoch完成后，查看`test_batch0_gt.jpg`查看batch 0 ground truth的labels。

查看`test_batch0_pred.jpg`查看test batch 0的预测。

训练的losses和评价指标被保存在Tensorboard和`results.txt`log文件。`results.txt`在训练结束后会被可视化为`results.png`。

```python
>>> from utils.utils import plot_results
>>> plot_results()
# 如果你是用远程连接请安装配置Xming: https://blog.csdn.net/akuoma/article/details/82182913
```

#### 7、推理过程

```python
$ python3 detect.py --source file.jpg  # image 
                            file.mp4  # video
                            ./dir  # directory
                            0  # webcam
                            rtsp://127.0.0.1/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            http://1127.0.0.1/PLTV/88888888/224/3221225900/1.m3u8  # http stream
````


```python
$ python3 detect.py --source /path/to/dataset/test/ --weights weights/best.pt --conf 0.1

$ python3 detect.py --source ./inference/images/ --weights weights/yolov5s.pt --conf 0.5

# inference  视频
$ python3 detect.py --source test.mp4 --weights weights/yolov5s.pt --conf 0.4
```




**Reference**

[1].https://github.com/ultralytics/yolov5

[2].https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
