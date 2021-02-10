# InstanceFusion

We present InstanceFusion, a robust real-time system to detect, segment, and reconstruct instance-level 3D objects of indoor scenes with a hand-held RGBD camera. It combines the strengths of deep learning and traditional SLAM techniques to produce visually compelling 3D semantic models. The key success comes from our novel segmentation scheme and the efficient instance-level data fusion, which are both implemented on GPU. Specifically, for each incoming RGBD frame, we take the advantages of the RGBD features, the 3D point cloud, and the reconstructed model to perform instance-level segmentation. The corresponding RGBD data along with the instance ID are then fused to the surfel-based models. In order to sufficiently store and update these data, we design and implement a new data structure using the OpenGL Shading Language. Experimental results show that our method advances the state-of-the-art (SOTA) methods in instance segmentation and data fusion by a large margin. Besides, our instance segmentation improves the precision of 3D reconstruction, especially in the loop closure. InstanceFusion system runs 20.5Hz on a consumer-level GPU, which supports a large number of augmented reality (AR) applications (e.g., 3D model registration, virtual interaction, AR map) and robot applications (e.g., navigation, manipulation, grasping). To facilitate future research and reproduce our system more easily, source code, data, and the trained model are released on Github (https://github.com/Fancomi2017/InstanceFusion).
## Publication  
Please cite this work if you make use of our system in any of your own endeavors:
* InstanceFusion: Real-time Instance-level 3D Reconstruction of Indoor Scenes using a Single RGBD Camera.
## 1.How to build it?  
### 1.1 Build InstanceFusion with tensorflow version Mask R-CNN
* Ubuntu 16.04(Though many other linux distros will work fine)  
* CMake  
* OpenGL  
* CUDA 8.0  
*  OpenNI2  
*  SuiteSparse  
*  Eigen  
*  zlib  
*  libjpeg  
*  opencv 3.1.0  
*  Pangolin  
*  libflann-dev  
*  Python3    
  
The system has been developed for Linux. It has been tested on Ubuntu 16.04, with gcc-5.4, CUDA 8.0, NVIDIA Driver 410.78, and CMake 3.5.1, with an Intel Core i7-4790 CPU @ 3.60GHz and Nvidia Geforce GTX 1080. Clone recursively with:  

    git clone --recursive https://github.com/Fancomi2017/InstanceFusion.git  
    
Make sure you can have all the dependencies and can compile and run the two major software packages this framework uses: elasticfusion and Mask-RCNN. They have both been slightly modified for this repo, so to be sure they are working build both of the projects that are cloned within the this repo. The changes however are minor, so the compilation and dependency instructions provided for those projects still apply.
Other than this, you have to build and install flann:  
  
 `cd /deps/flann-1.8.4/build`  
 `cmake ..`  
 `make -j8`  
 `sudo make install`   
  
Python virtual environment:  
  
    sudo -H pip3 install virtualenv  
    highlight "Setting up virtual python environment..."  
    virtualenv python-environment  
    source ~/python-environment/bin/activate  
    pip3 install pip --upgrade  
    pip3 install tensorflow-gpu==1.4.0  
    pip3 install scikit-image  
    pip3 install keras==2.0.8  
    pip3 install IPython  
    pip3 install h5py 
    pip3 install cython
    pip3 install imgaug  
    pip3 install opencv-python  
    pip3 install pytoml  
      
You can use this sentence " source ~/python-environment/bin/activate " to activate the python virtual environment.

If both of the dependencies are working, make a build directory and compile - this should build both sub-projects and then InstanceFusion.  
  
  `cd InstanceFusion`  
  `mkdir build`  
  `cd build`  
  `cmake ..`  
  `make -j8`   
    
Finally, you need to modify the path information of the Mask-RCNN network, such as open~ /Instancefusion/build/ mask_ori.py and Change MASK_RCNN_DIR to the path of your Mask-RCNN(~/Instancefusion/deps/mask_rcnn).  
### 1.2 Build InstanceFusion with PaddlePaddle version Mask R-CNN Benchmark
To get a faster segmentation speed, you can build InstanceFusion with PaddlePaddle version Mask R-CNN Benchmark. Please follow the instruction:  https://github.com/PaddlePaddle/models/tree/v1.3/fluid/PaddleCV/rcnn
## 2.Download Models
The Mask-RCNN model are available [here](https://github.com/matterport/Mask_RCNN/releases) with the mask_rcnn_coco.h5. Download and copy them to the Mask-RCNN subfolder of this project Instancefusion/deps/mask_rcnn. The Mask-RCNN Benchmark model are available [here](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_fbnet_xirb16d_dsmask.pth).
## 3.How to run it?  
If you have a kinect camera and OpenNI2 working (i.e. you can run ElasticFusion live) then you can run InstanceFusion classes by simply running the program with no arguments in the build directory. You need to make sure OpenNI2 can detect and access the feed from the camera.  
  
  `./InstanceFusion`  
  
You can test InstanceFusion on some dataset, such as  RGB-D Scenes Dataset v.2(RGB-D Scenes Dataset v.2 is available [here](http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/)) and dyson_lab.klg(available [here](https://www.doc.ic.ac.uk/~sleutene/datasets/elasticfusion/dyson_lab.klg)). After you download the RGB-D Scenes Dataset v.2, you can run dataset/transform.py to generate data.txt, such as a result in dataset/data.txt. To run on RGB-D Scenes Dataset v.2, you should provide a parameter to tell the program where data.txt is, note that the path should not contain spaces:
  
  `./InstanceFusion /path/to/your/rgbd-scenes-v2/imgs/scene_09/data.txt`  
  
And run on dyson_lab.klg like this:
  
  `./InstanceFusion -l /path/to/your/dyson_lab.klg`

## 4.License
InstanceFusion is freely available for non-commercial use only. Full terms and conditions which govern its use are in the LICENSE.txt file.
