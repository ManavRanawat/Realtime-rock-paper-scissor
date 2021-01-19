# -*- coding: utf-8 -*-
"""
Created on Mon May 25 01:59:12 2020

@author: Manav Ranawat
"""

import cv2
import os
import sys
import numpy as np

#make sure this variable has your current project directory
current_dir = os.getcwd() + '\\'
# print(current_dir)

cap = cv2.VideoCapture(0)

#label -> rock or paper or scissor
label = sys.argv[1]

SAVE_PATH = os.path.join(current_dir, label)
# print(SAVE_PATH)

try:
    os.mkdir(SAVE_PATH)
except FileExistsError:
    pass

#used for uniquely naming the images
img_counter = int(sys.argv[2])
max_img = sys.argv[3]
 

#code for clicking the image from camera
while(1):
    _,frame = cap.read()
    cv2.imshow('Collecting Data',frame[50:350,100:450])
    # if ' '(space bar) is pressed->image is captured
    if cv2.waitKey(1) & 0xFF == ord(' '):
        img_name = label+"{}.png".format(img_counter)   #naming each image uniquely
        cv2.imwrite(os.path.join(SAVE_PATH,img_name), frame)    #storing the image in your directory
        print("{} written!".format(img_name))
        img_counter += 1
    if img_counter>= int(max_img) :
        break

cap.release()
cv2.destroyAllWindows()