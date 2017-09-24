# -*- coding: utf-8 -*-
import os
import sys
# 添加python到环境变量
# caffe_root = '../../python'
# sys.path.insert(0, caffe_root)
import caffe
import numpy as np
import cv2
import scipy.io
from matplotlib import pyplot as plt

caffe.set_mode_gpu()
caffe.set_device(0)

proto_use = 'network/MSCNN_deploy.prototxt'
proto_file = 'network/MSCNN.prototxt'
model_file = 'caffemodel/mscnn_partA_iter_380000.caffemodel'
result_output =  'test_result_380000.txt'

# 能够使用edit_proto的主要原因是全卷积网络
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
	# print(estdmap.shape)
	# 1*1*height*width->height*width，只要元素总量一致即可
	estdmap = np.reshape(estdmap, (estdmap.shape[2], estdmap.shape[3]))
	count = int(np.ceil(np.sum(out['estdmap'])))
	return count

def final_result(images_path, ground_truth_path):
	if not images_path.endswith('/'):
		images_path += '/'
	if not ground_truth_path.endswith('/'):
		ground_truth_path += '/'

	images_list = os.listdir(images_path)
	num_test = len(images_list)
	predict_list = np.zeros((num_test,1))
	ground_truth_list = np.zeros((num_test, 1))
	mae = 0
	mse = 0

	with open(result_output, 'w') as result_file:
		for i in range(num_test):
			image_name = images_list[i]
			gt_name = 'GT_'+image_name[:-4]+'.txt'
			image_path = images_path + image_name
			gt_path = ground_truth_path + gt_name
			count = predict_count(image_path)
			with open(gt_path, 'rb') as f:
				gt = int(f.readline())
			predict_list[i] = count
			ground_truth_list[i] = gt
			result_file.write('predicting {0}, ground truth {1}\n'.format(count, gt))
			# print 'processing image %d, predicting: %d, ground truth: %d' % (i+1, count, gt)
		mae = np.mean(np.abs(predict_list - ground_truth_list))
		mse = np.sqrt(np.mean((predict_list - ground_truth_list)**2))
		result_file.write('The total MAE is {0}.\nThe total MSE is {1}.\n'.format(mae, mse))
	print 'The total MAE is ', mae
	print 'The total MSE is ', mse

if __name__ == '__main__':
	images_path = 'D:/Caffe/caffe-windows-ms/data/ShanghaiTech/part_A/test_data/images'
	ground_truth_path = 'D:/Caffe/caffe-windows-ms/data/ShanghaiTech/part_A/test_data/GT_NUM'
	final_result(images_path, ground_truth_path)
	


