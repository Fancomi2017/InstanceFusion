import os
import glob
import shutil
import cv2
import numpy as np
from PIL import Image

Path_to_Dataset = "/home/jun/Desktop/dataset/scene_09"
images = glob.glob(os.path.join(Path_to_Dataset, "*.png"))
images.sort()

if (os.path.isdir(Path_to_Dataset + '/depth')) == False:
    os.mkdir(Path_to_Dataset + '/depth')
if (os.path.isdir(Path_to_Dataset + '/rgb')) == False:
    os.mkdir(Path_to_Dataset + '/rgb')

tf = open(Path_to_Dataset + '/data.txt', "a")
name1 = ''
name2 = ''
i = 1
for im in images:
    str1 = im.split('/')
    str1 = str1[-1]
    isDepth = str1.split('-')
    if isDepth[-1] == 'depth.png':
        image = Image.open(im)
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                image.putpixel((x,y), image.getpixel((x,y))//10)
        image.save(im)
        shutil.move(im, Path_to_Dataset + '/depth')
        name1 = str1
    else:
        shutil.move(im, Path_to_Dataset + '/rgb')
        name2 = str1
    if name1 != '' and name2 != '':
        tf.write(str(i) +" ./depth/" + name1 + ' ' + "./rgb/" + name2 +" 0 0"+ '\n')
        name1 = ''
        name2 = ''
        i = i + 1
tf.close()

