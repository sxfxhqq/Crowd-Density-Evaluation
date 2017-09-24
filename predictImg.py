# -*- coding: utf-8 -*-
import os
import sys
import caffe
import numpy as np
import cv2
import scipy.io
from matplotlib import pyplot as plt # plt 用于显示图片
from matplotlib import cm as cm # cm 用于设置热力图
import matplotlib.image as img # img 用于读取图片

caffe.set_mode_gpu()
caffe.set_device(0)

proto_use = 'proto/MSCNN_deploy.prototxt'
proto_file = 'proto/MSCNN.prototxt'
model_file = 'snapshot_aug/mscnn_partA_iter_380000.caffemodel'
images_path = 'D:/Caffe/caffe-windows-ms/data/ShanghaiTech/part_A/test_data/images/IMG_15.jpg'

def edit_proto(img_height, img_width):
	with open(proto_file, 'r') as template_file:
		template_proto = template_file.read()
		# 写好测试模型文件，即复制MSCNN到MSCNN_deploy
		with open(proto_use, 'w') as out_file:
			out_file.write(template_proto.format(height=img_height, width=img_width))

def predict_count(image_path):
	frame = cv2.imread(image_path)
	if not frame.shape[2] == 1:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

def display(count,densityImg):
	fig = plt.figure(facecolor="w")
	img1 = fig.add_subplot(121)#121表示整体布局为1行2列，此为第1副
	originImg = img.imread(images_path)
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
	plt.show()   	

if __name__ == '__main__':
	count,densityImg = predict_count(images_path)
	# print(count)
	# print(estdmap)
	display(count,densityImg)
