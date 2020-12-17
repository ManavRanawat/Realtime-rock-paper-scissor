# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:06:05 2020

@author: Manav Ranawat
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import sys

#one hot encoding of each label
rps_to_vector= {
        'rock' : np.array([1,0,0]),
        'paper' : np.array([0,1,0]),
        'scissors' : np.array([0,0,1])
    }

# make sure this has your current directory path
current_dir = os.getcwd()
# path = SAVE_PATH = os.path.join(current_dir, sys.arg[1]) #path to which folder to go(like RPS)


imglist = list()

#labeling each data to its enconding vector
for dr in os.listdir(current_dir) :
    if dr not in ['rock','paper','scissors'] :
        continue
    # here dr will be the images folder(i.e. rock or paper or scissors)
    vec = rps_to_vector[dr]
    ctr = 0
    for img in os.listdir(os.path.join(current_dir,dr)) :
    	# make sure the path is correct
        dr_img = dr + "\\"
        path = os.path.join(current_dir,dr_img+img)
        # print(os.getcwd())
        pic = cv2.imread(path)
        # doing some data augmentation and creating more data
        pic = cv2.resize(pic,(300,300))
        imglist.append([pic,vec])
        imglist.append([cv2.flip(pic,1),vec])
        imglist.append([cv2.resize(pic[50:250,50:250],(300,300)),vec])
        # ctr = ctr + 3	#counting the number of image for each label(uncomment this and next line if you want.)
    # print(dr,ctr)

#shuffling our data randomly
np.random.shuffle(imglist)

img, label = map(list, zip(*imglist)) 

img = np.array(img)
label = np.array(label)

#Uncomment this if you want to see any image and its label. Your can play around and change the numbers ;)
# =============================================================================
# 
# cv2.imshow('img',imglist[-1][0])
# print(imglist[-1][1])
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
# 
# =============================================================================
        

#creating our model now
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing import image      
from keras.optimizers import Adam  
from keras.applications import DenseNet121
from keras.callbacks import EarlyStopping,ModelCheckpoint      

#using transfer learning on DenseNet121.(you can also use some other model if you want to(just checkout keras.application on keras documentation))
base_model = DenseNet121(include_top = False ,weights = 'imagenet',classes = 3,input_shape = (300,300,3))
base_model.trainable = True

model = Sequential()
model.add(base_model)
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(3,activation = 'softmax'))
model.compile(optimizer = Adam(),loss = 'categorical_crossentropy',metrics = ['acc'])

#storing the models weight in a file
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"

callback = [
    ModelCheckpoint(
        filepath, 
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True,
        mode='auto'
    ),
    EarlyStopping(patience = 2)
    ]


history = model.fit(
    x = img,
    y = label,
    batch_size = 32, 
    epochs = 4,
    callbacks = callback, 
    validation_split = 0.2
    )

#Uncomment this if you want to checkout how your model performed
# =============================================================================
# 
# preds = model.evaluate(x=img,y=label)
# print()
# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))
# 
# =============================================================================

#And finally, saving the model
model.save_weights('rps-model.h5')

with open("rps-model.json", "w") as json_file:
    json_file.write(model.to_json())



