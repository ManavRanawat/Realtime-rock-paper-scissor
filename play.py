# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:10:32 2020

@author: Manav Ranawat
"""
#importing libraries
import numpy as np
import os
import sys
from keras.models import model_from_json
import cv2
import random

#one hot enconding of the 3 classe
rps_to_vector= {
        'rock' : np.array([1,0,0]),
        'paper' : np.array([0,1,0]),
        'scissors' : np.array([0,0,1])
    }

# Getting the class name from one hot encoding as an input
arr_to_shape = {np.argmax(rps_to_vector[x]):x for x in rps_to_vector.keys()}


# reshaping input image according to the needed shape for our model
def prepImg(pth):
    return cv2.resize(pth,(300,300)).reshape(1,300,300,3)


# Make sure the path entered here is correct. Change it accordingly
curr_path = os.getcwd()


# loading the saved model. 
# NOTE: change the name of the .json and .h5 file accordingly
with open(os.path.join(curr_path,'model.json'), 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(os.path.join(curr_path,"weights-improvement-03-0.74.h5"))


# activating camera using openCV
cap = cv2.VideoCapture(0)


# Welcome page 
while True :
    
    ret,frame = cap.read()
    cv2.rectangle(frame, (0, 0), (800, 500), (255, 255, 255), -1)

    cv2.putText(frame,"Welcome to ROCK PAPER SCISSORS",(50,140),cv2.FONT_HERSHEY_SIMPLEX,1,(212,20,0),2,cv2.LINE_AA)

    cv2.putText(frame,"Enter Space to Play",(175,280),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),2,cv2.LINE_AA)
    
    cv2.imshow('Rock Paper Scissor',frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break


# dictionary that maps who wins against whom
to_win = {
        'rock' : 'scissors',
        'scissors' : 'paper',
        'paper' : 'rock'
    }


# incrementation of points according to what players play
def winner(pscore,bscore,player,bot) :
    if player == bot:
        return pscore,bscore
    elif to_win[player] == bot :
        return pscore+1,bscore
    else:
        return pscore,bscore+1

# Initializiation of values
pscore = 0 # players score
bscore = 0 # bots score
options = ['rock','paper','scissors']


############
# Comment '***' and uncomment '**' if you want the game to go on till someone wins fixed number of points
# Comment '**' and uncomment '***' if you want the game to end after fixed number of rounds
############
wins = True #**
towin = 2 #**
# Number of rounds(times) the game will continue
# n = 5 #***
############


# Game time!

# for i in range(n): # ***
while wins: #**
    # initialize bots move and prediction of the model
    bot = ""
    pred = ""
    
    # if anyone wins some fixed points then stop 
    if pscore == towin or bscore == towin : #**
        wins = False 
        break#**
            
    # after 150 iteration new round will begin.
    # NOTE : you can increase the oteration to increase the time taken for a round
    for t in range(150): 
        _,frame = cap.read()
        frame = cv2.flip(frame,1)
        
        if t<60:
            # 3..2..1 timer for the user.
            cv2.putText(frame,str((t//20)+1),(200,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        elif t < 70:
            # When the timer is 1. Model will predict your gesture from the frame
            pred = arr_to_shape[np.argmax(loaded_model.predict(prepImg(frame[50:350,100:400])))]
        elif t == 70:
            # bot playing its move randomly
            bot = random.choice(options)  
        elif t == 80:
            # updation of score accordingly
            pscore,bscore = winner(pscore,bscore,pred,bot)
            break
        
        # box where you have to do the hand gesture and model will predict using that frame
        cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
        # makind the other background white
        cv2.rectangle(frame, (0, 0), (800, 150), (255, 255, 255), -1)
        cv2.rectangle(frame, (0, 150), (100, 500), (255, 255, 255), -1)
        cv2.rectangle(frame, (300, 150), (800, 500), (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 350), (800, 500), (255, 255, 255), -1)
        
        # Displaying title
        cv2.putText(frame,"ROCK PAPER SCISSORS",(150,50),cv2.FONT_HERSHEY_SIMPLEX,1,(212,20,0),2,cv2.LINE_AA)
        
        # Displaying prediction of the model from the frame
        cv2.putText(frame,pred,(150,140),cv2.FONT_HERSHEY_SIMPLEX,1,(33,33,33),2,cv2.LINE_AA)
        # Displaying what bot played
        cv2.putText(frame,'Bot Played: {}'.format(bot),(320,140),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        # Displaying the updated score 
        cv2.putText(frame,'Player Score: {}    Bot Score: {}'.format(pscore,bscore),(100,400),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),2,cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissor',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(pred,bot)

# deciding the winner based on scores
win = ""
if pscore < bscore :
    win = "YOU LOSE!:("
elif pscore == bscore:
    win = "IT'S A DRAW!!"
else:
    win = "YOU WIN!!:)"
 
# After finishing the game. Showing results
while True :
    _,frame = cap.read()
    # color the background
    cv2.rectangle(frame, (0, 0), (800, 500), (255, 255, 255), -1)
    # writing results text
    cv2.putText(frame,win,(50,300),cv2.FONT_HERSHEY_SIMPLEX,3,(212,20,0),3,cv2.LINE_AA)
    # displaying players and bots final score
    cv2.putText(frame,'Player Score: {}    Bot Score: {}'.format(pscore,bscore),(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),2,cv2.LINE_AA)
    # exiting the game
    cv2.putText(frame,"Enter q to Exit",(200,400),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),1,cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# deactivating and closing the camera and tabs  
cap.release()
cv2.destroyAllWindows()

