## 一、准备阶段
该模型是由openpose改编的，事先需要下载：

 - 1、训练好的pretrained network模型文件（[pretrained weight
   download](https://www.dropbox.com/s/xh5s7sb7remu8tx/openpose_coco.npy?dl=0)）：`openpose_coco.npy
   · 199.56 MB`
 - 2、Tensorflow Graph File图文件（CMU's model）：  

```
$ cd models/graph/cmu
$ bash download.sh
```
当然如果要批量处理起来可能需要另外写一个。

## 二、训练与预测
#### 2.1 关于训练 
一般是coco数据集训练而得，有18个关键点。训练的过程在原内容中已经封装的非常好，所以笔者最开始想借鉴，但是发现封的太死，要命了！
后来发现有一个小哥在参加完 AI challenger挑战赛，把训练过程简单写了出来，可参考：[galaxy-fangfang/AI-challenger-Realtime_Multi-Person_Pose_Estimation-training](https://github.com/galaxy-fangfang/AI-challenger-Realtime_Multi-Person_Pose_Estimation-training) 

关键点的关联可见下图：
![这里写图片描述](https://camo.githubusercontent.com/5833ee83e638a1b622a16fd6447d64a9668efcf5/687474703a2f2f696d672e626c6f672e6373646e2e6e65742f32303137303930383134343233383631303f77617465726d61726b2f322f746578742f6148523063446f764c324a736232637559334e6b626935755a5851766147467763486c6f62334a70656d6c7662673d3d2f666f6e742f3561364c354c32542f666f6e7473697a652f3430302f66696c6c2f49304a42516b46434d413d3d2f646973736f6c76652f37302f677261766974792f536f75746845617374)

#### 2.2 关于预测
预测使用的命令行模式：

```
python3 run.py --model=mobilenet_thin --resolution=432x368 --image=...
```
笔者简单的从原作者的预测命令中提取出了相关预测信息，自己简单写了两个函数：`get_keypoint、PoseEstimatorPredict`在[predict.py](https://github.com/mattzheng/tf-pose-estimation-applied/blob/master/predict/predict.py)之中

目前支持两种模型：`mobilenet_thin`以及`cmu`
其中model的类型一共六种，具体可见文档：`/src/network.py`

其中，`PoseEstimatorPredict`两种输出方式：

 - plot=False，返回一个内容：关键点信息
 - plot=True，返回两个内容：关键点信息+标点图片matrix

