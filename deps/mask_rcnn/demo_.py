#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
import time
import copy

get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

# In[5]:


# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
file_names = next(os.walk(IMAGE_DIR))[2]
file_names.sort()
images = []
results = []
#for i in range(len(file_names)):
for i in range(2):
    print(file_names[i])
    file_names[i] = os.path.join(IMAGE_DIR, file_names[i])
    image = skimage.io.imread(file_names[i])
    images.append(image)
    # Run detection
    result = model.detect([image], verbose=1)
    results.append(result[0])
#results = model.detect(images, verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(images[0], r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])


# In[6]:


# In[6]:


MAX_INSTANCE = 200
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

MIOU_THRESHOLD = 0.30
WINDOWEDGE_THRESHOLD1 = 100
WINDOWEDGE_THRESHOLD2 = 640
def testInstanceExist(mask_id,result,result_pre,prob):
    if result_pre == None:
        return -1
    for i in range(result_pre['masks'].shape[2]):
        # ??????  first detect is choose
        #if result['class_ids'][mask_id] != result_pre['class_ids'][i] :
        #    continue
        test = maskmIOU(result['masks'][:,:,mask_id],result_pre['masks'][:,:,i])
        if test > MIOU_THRESHOLD:
            return result_pre['instance_id'][i]
    return -1


def maskmIOU(mask1,mask2):
    intersection = 0
    union = 0
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i][j] == 1 or mask2[i][j] == 1:
                union = union + 1
            if mask1[i][j] == 1 and mask2[i][j] == 1:
                intersection = intersection + 1
    return intersection/union

def updateProb(mask,scores,prob,instance_id):
    #print(mask.shape)
    #print(prob.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 1:
                prob[0][instance_id][i][j] = scores

def maskCheckWindowEdge(mask):
    num = 0;
    for i in range(mask.shape[0]):
        if mask[i][5] == 1:
            num = num + 1
        if mask[i][mask.shape[1]-6] == 1:
            num = num + 1
    for j in range(mask.shape[1]):
        if mask[5][j] == 1:
            num = num + 1
        if mask[mask.shape[0]-6][j] == 1:
            num = num + 1
    #print(num)
    if num > WINDOWEDGE_THRESHOLD1 and num < WINDOWEDGE_THRESHOLD2:
        return 1
    else:
        return 0


instance_table = np.zeros((MAX_INSTANCE,4), dtype=np.int)      #整个视频序列的实例，对应类和颜色表
instance_num = 0

for i, result in enumerate(results):
    #显示节点
    print(i)
    T = time.time()
    print(time.time()-T)
    #加入实例项
    instance_id = np.zeros(result['masks'].shape[2], dtype=np.int);
    result.update({"instance_id": instance_id})
    #加入前图结果
    result_pre1 = None
    result_pre2 = None
    result_pre3 = None
    if i > 0 :
        if results[i-1]['masks'].shape[0] != 0:
            result_pre1 = results[i-1]
    if i > 1 :
        if results[i-2]['masks'].shape[0] != 0:
            result_pre2 = results[i-2]
    if i > 2 :
        if results[i-3]['masks'].shape[0] != 0:
            result_pre3 = results[i-3]
    #对于每一张图片
    #print("scores "+str(i) +" "+str(result['masks'].shape))
    prob = np.zeros((1,MAX_INSTANCE,WINDOW_HEIGHT,WINDOW_WIDTH));
    
    #No result
    if result['masks'].shape[0] == 0:
        continue
    for mask_id in range(result['masks'].shape[2]):
        #对于图片里每一个mask
        # 输出mask图
        #skimage.io.imsave(os.path.join(IMAGE_DIR, 'prob','mask_'+str(i)+'_'+str(mask_id)+'.png'), result['masks'][:,:,mask_id])
        
        #检查窗口边缘
        if maskCheckWindowEdge(result['masks'][:,:,mask_id]):
            result['instance_id'][mask_id] = -1
            continue
        #检查前几帧，判断相同实例
        instance = testInstanceExist(mask_id,result,result_pre1,prob)
        if instance == -1:
            instance = testInstanceExist(mask_id,result,result_pre2,prob)
        if instance == -1:
            instance = testInstanceExist(mask_id,result,result_pre3,prob)
        #注册全新实例
        if instance == -1:
            instance_table[instance_num][0] = result['class_ids'][mask_id]
            instance_table[instance_num][1] = random.randint(0,255)
            instance_table[instance_num][2] = random.randint(0,255)
            instance_table[instance_num][3] = random.randint(0,255)
            instance = instance_num
            instance_num = instance_num + 1
        result['instance_id'][mask_id] = instance
        updateProb(result['masks'][:,:,mask_id],result['scores'][mask_id],prob,instance)

    print(time.time()-T)
    # 保存prob
    prob_text = os.path.join(IMAGE_DIR, 'prob','prob_'+str(i)+'.txt')
    f1 = open(prob_text, "w")
    f1.write("channel\ty\tx\tprob\tnum\n")
    for channel in range(MAX_INSTANCE):
        for j in range(WINDOW_HEIGHT):
            state = 0
            for k in range(WINDOW_WIDTH):
                if prob[0][channel][j][k] != 0:
                    if state == 0:
                        f1.write("%d\t%d\t%d\t%0.4f\t" % (channel,j,k,prob[0][channel][j][k]))
                        state = 1
                        count = 1
                    else:
                        count = count + 1
                else:
                    if state == 1:
                        f1.write("%d\n" % (count))
                        state = 0
            if state == 1:
                f1.write("%d\n" % (count))
                state = 0
    f1.close()
    print(time.time()-T)
    
    
    # 保存可视化图片
    visuImage = copy.deepcopy(images[i])
    for j in range(WINDOW_HEIGHT):
        for k in range(WINDOW_WIDTH):
            color = [0,0,0]
            flag = 0
            for mask_id in range(result['masks'].shape[2]):
                #排除不合格的mask
                if result['instance_id'][mask_id] == -1:
                    continue
                if result['masks'][j][k][mask_id] == 1:
                    color[0] += instance_table[result['instance_id'][mask_id]][1]
                    color[1] += instance_table[result['instance_id'][mask_id]][2]
                    color[2] += instance_table[result['instance_id'][mask_id]][3]
                    flag = flag + 1
            if flag == 0:
                continue
            color[0] = color[0] / flag
            color[1] = color[1] / flag
            color[2] = color[2] / flag
            visuImage[j][k][0] = (visuImage[j][k][0] + color[0])/2
            visuImage[j][k][1] = (visuImage[j][k][1] + color[1])/2
            visuImage[j][k][2] = (visuImage[j][k][2] + color[2])/2
    skimage.io.imsave(os.path.join(IMAGE_DIR, 'prob','visualize_'+str(i)+'.png'), visuImage)
    print(time.time()-T)
    

# 保存 instanceTable
instance_text = os.path.join(IMAGE_DIR, 'prob','instanceTable.txt')
f2 = open(instance_text, "w")
for i in range(instance_num):
    f2.write("class:%d R:%d G:%d B:%d\n" % (instance_table[i][0],instance_table[i][1],instance_table[i][2],instance_table[i][3]))
f2.close()


