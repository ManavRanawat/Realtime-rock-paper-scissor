# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:10:32 2020

@author: Manav Ranawat
"""

import numpy as np
import os
import sys
from keras.models import model_from_json
import cv2
import random

rps_to_vector= {
        'rock' : np.array([1,0,0]),
        'paper' : np.array([0,1,0]),
        'scssors' : np.array([0,0,1])
        # 'others' : np.array([0,0,0])
    }
arr_to_shape = {np.argmax(rps_to_vector[x]):x for x in rps_to_vector.keys()}

def prepImg(pth):
    return cv2.resize(pth,(300,300)).reshape(1,300,300,3)

curr_path = os.getcwd()+'\\Desktop\\projects\\RPS'

with open(os.path.join(curr_path,'rps-model.json'), 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(os.path.join(curr_path,"weights-improvement-03-0.74.h5"))


cap = cv2.VideoCapture(0)

# label : play

while True :
    
    ret,frame = cap.read()
    cv2.rectangle(frame, (0, 0), (800, 500), (255, 255, 255), -1)

    cv2.putText(frame,"Welcome to ROCK PAPER SCISSORS",(50,140),cv2.FONT_HERSHEY_SIMPLEX,1,(212,20,0),2,cv2.LINE_AA)

    cv2.putText(frame,"Enter Space to Play",(175,280),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),2,cv2.LINE_AA)
    # pred = arr_to_shape[np.argmax(loaded_model.predict(prepImg(frame[50:350,100:400])))]
    
    
    # cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
    # frame = cv2.putText(frame,pred,(150,140),cv2.FONT_HERSHEY_SIMPLEX,1,(250,250,0),2,cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor',frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

n = 3
to_win = {
        'rock' : 'scssors',
        'scssors' : 'paper',
        'paper' : 'rock'
    }

def winner(pscore,bscore,player,bot) :
    if player == bot:
        return pscore,bscore
    elif to_win[player] == bot :
        return pscore+1,bscore
    else:
        return pscore,bscore+1

pscore = 0
bscore = 0
options = ['rock','paper','scssors']

wins = True
towin = 3
# for i in range(n):
while wins:
    bot = ""
    pred = ""
    for t in range(150):
        _,frame = cap.read()
        if t<60:
            cv2.putText(frame,str((t//20)+1),(200,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        elif t < 70:
            pred = arr_to_shape[np.argmax(loaded_model.predict(prepImg(frame[50:350,100:400])))]
        elif t == 70:
            bot = random.choice(options)  
        elif t == 80:
            pscore,bscore = winner(pscore,bscore,pred,bot)
            break
        cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (800, 150), (255, 255, 255), -1)
        cv2.rectangle(frame, (0, 150), (100, 500), (255, 255, 255), -1)
        cv2.rectangle(frame, (300, 150), (800, 500), (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 350), (800, 500), (255, 255, 255), -1)
        cv2.putText(frame,"ROCK PAPER SCISSORS",(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(212,20,0),2,cv2.LINE_AA)
        cv2.putText(frame,pred,(150,140),cv2.FONT_HERSHEY_SIMPLEX,1,(33,33,33),2,cv2.LINE_AA)
        cv2.putText(frame,'Bot Played: {}'.format(bot),(320,140),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame,'Player Score: {}    Bot Score: {}'.format(pscore,bscore),(100,400),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),2,cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissor',frame)
        if pscore == towin or bscore == towin :
            wins = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(pred,bot)

win = ""
if pscore < bscore :
    win = "YOU LOSE!:("
elif pscore == bscore:
    win = "IT'S A DRAW!!"
else:
    win = "YOU WIN!!:)"
    
while True :
    _,frame = cap.read()
    cv2.rectangle(frame, (0, 0), (800, 500), (255, 255, 255), -1)
    cv2.putText(frame,win,(50,300),cv2.FONT_HERSHEY_SIMPLEX,3,(212,20,0),3,cv2.LINE_AA)
    cv2.putText(frame,'Player Score: {}    Bot Score: {}'.format(pscore,bscore),(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Enter q to Exit",(200,400),cv2.FONT_HERSHEY_SIMPLEX,1,(212,129,0),1,cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cv2.waitKey(1) & 0xFF == ord('r'):
        # goto play


  
cap.release()
cv2.destroyAllWindows()

