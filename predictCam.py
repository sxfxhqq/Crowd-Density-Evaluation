# -*- coding: utf-8 -*-
import os
import sys
import caffe
import numpy as np
import cv2
import scipy.io
import time
from matplotlib import pyplot as plt # plt 用于显示图片
from matplotlib import cm as cm # cm 用于设置热力图
import matplotlib.image as img # img 用于读取图片

interval = 2       	# 捕获图像的间隔，单位：秒
num_frames = 1    	# 捕获图像的总帧数
out_fps = 24        	# 输出文件的帧率

caffe.set_mode_gpu()
caffe.set_device(0)

proto_use = 'proto/MSCNN_deploy.prototxt'
proto_file = 'proto/MSCNN.prototxt'
model_file = 'snapshot_aug/mscnn_partA_iter_380000.caffemodel'

# VideoCapture(0)表示打开默认的相机
cap = cv2.VideoCapture(0)
# 获取捕获的分辨率
size =(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
	int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 设置要保存视频的编码，分辨率和帧率
video = cv2.VideoWriter(
	"crowd.avi", 
	cv2.VideoWriter_fourcc('M','P','4','2'), #指定了视频编码的格式，此处用的是MP42，也就是MPEG-4
	out_fps, 
	size
)

def edit_proto(img_height, img_width):
	with open(proto_file, 'r') as template_file:
		template_proto = template_file.read()
		# 写好测试模型文件，即复制MSCNN到MSCNN_deploy
		with open(proto_use, 'w') as out_file:
			out_file.write(template_proto.format(height=img_height, width=img_width))

def predict_count(frame):
	img_height = frame.shape[0]
	img_width = frame.shape[1]
	frame = np.reshape(frame, (img_height, img_width, 1))
	edit_proto(img_height, img_width)
	net_full_conv = caffe.Net(proto_use, model_file, caffe.TEST)
	transformers = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
	transformers.set_transpose('data', (2,0,1))
	# 转化为数组再传入网络
	out = net_full_conv.forward_all(data=np.array([transformers.preprocess('data', frame)]))
	# 输出密度图，（图片数量,通道,高度,宽度）
	estdmap = out['estdmap']
	print(estdmap.shape)
	# 1*1*height*width->height*width，只要元素总量一致即可
	estdmap = np.reshape(estdmap, (estdmap.shape[2], estdmap.shape[3]))
	# sum为求和，ceil为向下取整
	count = int(np.ceil(np.sum(out['estdmap'])))
	return count,estdmap

def Img2Map(originImg,count,densityImg):
	fig = plt.figure(facecolor="w")
	img1 = fig.add_subplot(121)#121表示整体布局为1行2列，此为第1副
	img1.imshow(originImg)
	# plt.axis('off') # 不显示坐标轴
	plt.title("Original Image",fontweight='bold')
	plt.xlabel('Total number is '+str(count),fontweight='bold')

	img2 = fig.add_subplot(122)#121表示整体布局为1行2列，此为第2副
	cmap = cm.get_cmap('rainbow')# 设置cmap形式，对于同一幅图，彩虹图形象地显示密集的地方，不同图像不行
	map = img2.imshow(densityImg, cmap=cmap) # 显示图片，默认为热力图
	bar=plt.colorbar(mappable=map,fraction=0.026, pad=0.04)
	# bar.set_label('over 0.15 indicates crowdy',fontweight='bold')
	# plt.axis('off') # 不显示坐标轴
	plt.title("Density Map",fontweight='bold')
	plt.xlabel('More than 0.1 means crowded',fontweight='bold')

	plt.savefig('test.png')
	# plt.show()

if __name__ == '__main__':

	# 对于一些低画质的摄像头，前面的帧可能不稳定，略过
	for i in range(42):
		cap.read()#没有返回即放弃

	# 开始捕获，通过read()函数获取捕获的帧
	try:
		for i in range(num_frames):
			_, frame = cap.read()#opencv读取图片格式为BGR
			count,estdmap=predict_count(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			Img2Map(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),count,estdmap)
			time.sleep(interval/2)	# 进程挂起，若要显示图片需要多进程
			# Figure = cv2.cvtColor(cv2.imread('test.png'),cv2.COLOR_BGR2RGB)
			Figure = cv2.imread('test.png')
			video.write(Figure)

			# 如果希望把每一帧也存成文件，比如制作GIF，则取消下面的注释
			filename = '{:0>6d}.png'.format(i)#0>6d表示6位数
			cv2.imwrite(filename, Figure)

			print('Frame {} is captured.'.format(i))
			time.sleep(interval/2) # 进程挂起，若要显示图片需要多进程
	except KeyboardInterrupt:#用来获取用户Ctrl+C的中止
		# 提前停止捕获
		print('Stopped! {}/{} frames captured!'.format(i, num_frames))

	# 释放资源并写入视频文件
	video.release()
	cap.release()