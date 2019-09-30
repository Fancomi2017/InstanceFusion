# # From PaddleCV_rcnn_infer 
# 
import os
import time
import numpy as np
import sys
import cv2

MASKRCNN_PADDLE_DIR = "/media/user/Disk1/penghaotian/PaddlePaddle/rcnn"
sys.path.insert(0, MASKRCNN_PADDLE_DIR)

from eval_helper import *
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args
import models.model_builder as model_builder
import models.resnet as resnet
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
from roidbs import DatasetPath

from data_utils import prep_im_for_blob
import skimage.io
import pycocotools.mask as mask_util

# Class Names
# same as mask_ori
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

#config.py & (main in utility.py)
args = parse_args()
print_arguments(args)

#Step 1
image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
class_nums = cfg.class_num


#Step 2
model = model_builder.RCNN(add_conv_body_func=resnet.add_ResNet50_conv4_body,
				add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
				use_pyreader=False,
				mode='infer')
model.build_model(image_shape)
model_pred_boxes = model.eval_bbox_out()
model_masks = model.eval_mask_out()
place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

#Step 3
def if_exist(var):
	return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
fetch_list = [model_pred_boxes, model_masks]

# Global variables (used to communicate with c++)
masks = None
class_ids = None


def detect(im):
	global masks
	global class_ids

	#Step 4 input
	#path = "/media/user/Disk1/penghaotian/image_instance/graph-canny-segm-master/00570-color.png"
	#im = cv2.imread(path)
	im_height_ori = im.shape[0]
	im_width_ori  = im.shape[1]

	im, im_scale = prep_im_for_blob(im, cfg.pixel_means, cfg.TEST.scales[0], cfg.TEST.max_size)

	im_height = np.round(im_height_ori * im_scale)
	im_width = np.round(im_width_ori * im_scale)
	im_info = np.array([im_height, im_width, im_scale], dtype=np.float32)
	data = [(im, im_info)]
	
	#print(im_height_ori,im_width_ori)
	#print(im_height,im_width)

	#Step 5 run
	result = exe.run(fetch_list=[v.name for v in fetch_list],
					feed=feeder.feed(data),
					return_numpy=False)
	segms_out = segm_results(result[0], result[1], [data[0][1]])


	#Step 6 output
	results = []
	for dt in np.array(segms_out):
		segm, num_id, score = dt.tolist()
		if score < cfg.draw_threshold:
			continue
		maskOri = mask_util.decode(segm) * 255
		if maskOri.shape[0] == im_height_ori and maskOri.shape[1] == im_width_ori:
			maskNP = np.zeros([maskOri.shape[0],maskOri.shape[1]], np.uint8)
			maskNP[maskOri != 0] = 255 #visualize 255
			results.append([maskNP,num_id])
			del maskNP
		del maskOri
	
	#Step 6 to InstanceFusion
	results = sorted(results,key=lambda x:np.sum(x[0]),reverse = True)
	masks = []
	class_ids = []
	for r in results:
		masks.append(r[0])
		class_ids.append(r[1])

	del im_height_ori
	del im_width_ori
	del im
	del im_scale
	del im_height
	del im_width

	del im_info
	del data
	del result
	del segms_out
	del results



