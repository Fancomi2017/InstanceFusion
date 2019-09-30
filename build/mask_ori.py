#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# 预训练模型和图像分割

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io


MASK_RCNN_DIR = "/media/jun/Data/YangJun/Instancefusion/Instancefusion/deps/mask_rcnn"
sys.path.insert(0, MASK_RCNN_DIR)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.3	#GPU: 12G * 0.3 > 2.07G

config.gpu_options.visible_device_list="0"
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session, clear_session, _SESSION
set_session(session)

import coco
import utils
import model as modellib
import time
import copy


class InferenceConfig(coco.CocoConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# Class Names
# 这里用到的类编号和COCO的不同，比如COCO中不包含71号，且'teddy bear'为88，而这里'teddy bear'为78
# BG为0，person为1，...
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',		#5
               'bus', 'train', 'truck', 'boat', 'traffic light',			#10
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',		#15
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',		#22
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',		#28
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',		#33
               'kite', 'baseball bat', 'baseball glove', 'skateboard',			#37
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',		#42
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',			#48
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',		#54
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',		#60
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',		#66
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',		#71
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',		#77
               'teddy bear', 'hair drier', 'toothbrush']				#80

ROOT_DIR = os.getcwd()
MASK_DIR = MASK_RCNN_DIR
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "temp")

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# 权重
#COCO_MODEL_PATH = os.path.join(MASK_DIR, "mask_rcnn_coco_R50.h5")
COCO_MODEL_PATH = os.path.join(MASK_DIR, "mask_rcnn_coco.h5")
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Global variables (used to communicate with c++)
masks = None
class_ids = None


# Run Object Detection
def detect(rgb_image):
	global masks
	global class_ids
	
	#print(type(rgb_image))
	#print(rgb_image.shape)
	#print("Start")
	#A
	#start = time.time()
	result = model.detect([rgb_image], verbose=0) #1 show 0 not show
	result = result[0]
	#end = time.time()
	#print(end-start)

	#B
	#start = time.time()
	results = []	
	for mask_id in range(result['masks'].shape[2]):
		maskOri = result['masks'][:,:,mask_id]
		if maskOri.shape[0] == rgb_image.shape[0] and maskOri.shape[1] == rgb_image.shape[1]:
			maskNP = np.zeros([maskOri.shape[0],maskOri.shape[1]], np.uint8)
			maskNP[maskOri == 1] = 255 #visualize 255
			results.append([maskNP,result['class_ids'][mask_id]])
			del maskNP
		del maskOri
	#end = time.time()
	#print(end-start)

	#C
	#start = time.time()
	results = sorted(results,key=lambda x:np.sum(x[0]),reverse = True)
	masks = []
	class_ids = []
	for r in results:
		masks.append(r[0])
		class_ids.append(r[1])
	#end = time.time()
	#print(end-start)
