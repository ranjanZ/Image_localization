from camera_matrix import *

import os
import cv2
from localization import *
import sys
import matplotlib.pyplot as plt



K=get_K()
img_dir="../data/KITTI_sample/images/"
images=[img_dir+d for d in os.listdir(img_dir)]


img_loc=localization()
img1=cv2.imread(images[0], 0)
img2=cv2.imread(images[1], 0)
img_loc.processFirstFrame(img1)
img_loc.process2ndFrame(img2)
for i,img_path in enumerate(images):

    img = cv2.imread(img_path, 0)   #read in grayscale
    if(i>2):
        img_loc.processDefaultFrame(img)
        #print(img_loc.cur_R,img_loc.cur_t[:,0])
        angle=rotationMatrix2Angle(img_loc.cur_R)
        print(angle)

