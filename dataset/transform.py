import os
import glob
import shutil

Path_to_Dataset = "/home/jun/Desktop/dataset/scene_09"
images = glob.glob(os.path.join(Path_to_Dataset, "*.png")) # images类型是list，每一个元素都存储的是图片的完整路径
images.sort()
picPath = images[0].split('/')
picPath = picPath[:-1]
folderPath = ''
for s in picPath:
    folderPath = folderPath + s + '/'

# 创建depth与rgb文件夹
if (os.path.isdir(Path_to_Dataset + '/depth')) == False:
    os.mkdir(Path_to_Dataset + '/depth')
if (os.path.isdir(Path_to_Dataset + '/rgb')) == False:
    os.mkdir(Path_to_Dataset + '/rgb')

# 创建data.txt文件，移动文件到depth与color文件夹里
tf = open(Path_to_Dataset + '/data.txt', "a")
name1 = ''
name2 = ''
i = 1
for im in images:
    str1 = im.split('/')
    str1 = str1[-1] # 变为str类型，图片名字
    isDepth = str1.split('-')
    # depth图片放入depth文件夹
    if isDepth[-1] == 'depth.png':
        #os.rename(im, Path_to_Dataset + '/' + str(i) + '.png')
        shutil.move(im, Path_to_Dataset + '/depth') # 移动制定问价到目标文件夹
        name1 = str1
    else:
        #os.rename(im, Path_to_Dataset + '/' + str(i) + '.png')
        shutil.move(im, Path_to_Dataset + '/rgb')
        name2 = str1
    if name1 != '' and name2 != '':
        #tf.write(str(i) +" ./depth/" + name1 + ' ' + "./rgb/" + name2 +" 0 0"+ '\n')
        tf.write(str(i) +" ./depth/" + name1 + ' ' + "./rgb/" + name2 +" 0 0"+ '\n')
        name1 = ''
        name2 = ''
        i = i + 1
tf.close()

