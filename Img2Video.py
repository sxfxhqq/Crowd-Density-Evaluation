# -*- coding: utf-8 -*-
import cv2
import os
out_fps = 24        	# 输出文件的帧率
images_path = './resultImg'
images_list = os.listdir(images_path)
imglength = len(images_list)
video = cv2.VideoWriter(
	"./resultVideo/crowd.avi", 
	cv2.VideoWriter_fourcc('M','P','4','2'), #指定了视频编码的格式，此处用的是MP42，也就是MPEG-4
	out_fps, 
	(640,480)
)
for i in range(imglength):
	Figure = cv2.imread(images_path+'/'+'{:0>6d}.png'.format(i))
	video.write(Figure)
	print('Frame {} is captured.'.format(i))

video.release()
print('OK!')