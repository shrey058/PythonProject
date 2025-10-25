from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('PYTHON PROJRCT/haarcascade_frontalface_deafault.xml')

with open('data/names.pkl', 'rb')as f:
        LABELS=pickle.load(names,f)
with open('data/names.pkl', 'rb')as f:
        FACES=pickle.load(names,f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)

COL_NAMES = ['NAME', 'TIME']


while True:
    ret,frame=video.read()
    gray=cv2.cvtcolor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output= knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),1)
        cv2.rectangle(frame,(x,y), (x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40), (x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)

        cv2.rectangle(frame, (x,y),(x+w,y+h), (50,50,255), 1)
        attendance=[str(output[0]), str(timestamp)]
    cv2.imshow("Frame",frame)
    k=cv2.waitkey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

