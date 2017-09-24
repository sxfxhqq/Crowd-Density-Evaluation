# Crowd-Density-Evaluation
电子综合设计课程项目-饭堂人群密度检测，得益于任课导师的帮助才顺利完成项目，现将资料进行整理，借此总结所学知识。

### 概括工作
寻找合适的网络实现人群密度估计，终选用MSCNN作为核心算法，由于笔记本显存无法训练，使用邢老师实验室提供的caffemodel，编写应用代码，预测出人群密度的二维数组，绘制热力图并生成视频。在与他人交流过程，发现网络值得优化的地方：loss方程、kernelsize等。

### 知识准备
图像处理：《数字图像处理与机器视觉》《数字图像处理》

编程工具：python,opencv library,caffe(on windows)

参考文章：[人群密度检测-博客](http://blog.csdn.net/qq_14845119/article/category/6835127 "人群密度检测-博客")
、[人群密度检测-张营营](http://chuansong.me/n/443237851736 "人群密度检测-张营营")

### OpenCV算法
最初为方便，选择使用传统方法采集图像中的人体特征，统计人群数量，代码依赖于OpenCV。
#### Haar
Haar特征也称Haar-like特征，是一种简单且高效的图像特征，基于矩形区域相似的强度差异Haar小波。Haar特征的特点为：高类的可变性；低类的可变性；而向局部的强度差异；多尺度不变性；计算效率高。参考文章：[HOG特征&LBP特征&Haar特征](http://dataunion.org/20584.html)

#### 级联
OpenCV在物体检测上使用的是基于haar特征的级联表，级联将人脸检测过程拆分成了多个过程。在每一个图像小块中只进行一次粗略的测试。如果测试通过，接下来进行更详细的细节测试，依次重复。检测算法中有30至50个这种过程或者级联，只有在所有过程成功后才会最终识别到人脸。

#### 分类器
人们采用样本的haar特征训练出分类器，级联成完整的boost分类器，实现时分类器即数据组成的XML文件，OpenCV也自带了一些已经训练好的包括人眼、人脸和人体的分类器（位于OpenCV安装目录\data\haarcascades目录下，分类器是XML类型的文件）。参考文章：[浅析人脸检测之Haar分类器方法](http://www.cnblogs.com/ello/archive/2012/04/28/2475419.html)

#### 实现
使用python更加方便简洁，硬件选择PC或者Raspberry Pi，算法将读取图片并标注方框在人脸特征，改换haarcascades可用于其余特征。参考文章：[25行Python代码完成人脸识别](https://python.freelycode.com/contribution/detail/36)、[借助摄像头在Python中实现人脸检测](https://python.freelycode.com/contribution/detail/37)，完整代码在opencvdetect文件夹。

### MSCNN算法
MSCNN参考googlenet的inception结构而利用多尺度卷积核群，提取图像中丰富的人群密度信息，并使用全卷积网络直接得到人群密度图。MSCNN主要的优点是提取多尺度特征、单列网络参数少且易于训练，具体思想与优点参考论文《MULTI-SCALE CONVOLUTIONAL NEURAL NETWORKS FOR CROWD COUNTING》，网络结构可见 [MSCNN结构图](http://ethereon.github.io/netscope/#/gist/f7cd1ebe4319fc80dc8cc27827e097f4)，整个网络主要使用上海科技大学张营营其实验室的人群密度数据集，此处有训练好的caffemodel。

文件说明：
- predictCam.py用于抓取PC摄像头拍摄外界并进行人群密度估计
- predictImg.py用于对单张图片进行人群密度估计
- 目前提供的文件仍不足以运行代码，原因是缺少CrowdData层源码，这份源码属于导师不敢擅自公开

