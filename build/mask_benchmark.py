#!/usr/bin/env python
# coding: utf-8

# # maskrcnn_benchmark Demo
# 

import os
import sys
MASKRCNN_BENCHMARK_DIR = "/media/jun/Data/YangJun/Instancefusion/Instancefusion/deps/maskrcnn-benchmark-master"
sys.path.insert(0, MASKRCNN_BENCHMARK_DIR)

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import time
import skimage.io

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

#config_file = os.path.join(MASKRCNN_BENCHMARK_DIR , "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
config_file = os.path.join(MASKRCNN_BENCHMARK_DIR , "configs/e2e_mask_rcnn_fbnet_xirb16d_dsmask_600.yaml")


# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


coco_demo = COCODemo(
    cfg,
    min_image_size=512,
    confidence_threshold=0.7,
)



# Global variables (used to communicate with c++)
masks = None
class_ids = None


# Run Object Detection
def detect(rgb_image):
	global masks
	global class_ids
	
	#ok
	#skimage.io.imsave('./temp/visualization.png', rgb_image)
	
	#PART A
	predictions = coco_demo.compute_prediction(rgb_image)
	top_predictions = coco_demo.select_top_predictions(predictions)

	# Class Names
	# 这里用到的类编号同mask_ori
	#bbox   = top_predictions.get_field('bbox').numpy()
	labels = top_predictions.get_field('labels').numpy()
	maskP   = top_predictions.get_field('mask').numpy()

	#PART B
	results = []
	for mask_id in range(len(top_predictions)):
		maskOri = maskP[mask_id,0,:,:]
		if maskOri.shape[0] == rgb_image.shape[0] and maskOri.shape[1] == rgb_image.shape[1]:
			maskNP = np.zeros([maskOri.shape[0],maskOri.shape[1]], np.uint8)
			maskNP[maskOri != 0] = 255 #visualize 255
			results.append([maskNP,labels[mask_id]])
			del maskNP
		del maskOri



	#PART C
	results = sorted(results,key=lambda x:np.sum(x[0]),reverse = True)
	masks = []
	class_ids = []
	for r in results:
		masks.append(r[0])
		class_ids.append(r[1])

	del results
	del labels
	del maskP
	del predictions
	del top_predictions
	del rgb_image
	
