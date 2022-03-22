#coding:utf-8

import cv2

import os


cPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cPath)


video_capture = cv2.VideoCapture(0)


while True:

    # Capturez image par image
    ret, frames = video_capture.read()


    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(30, 30),

        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Dessinez un rectangle autour des visages

    for (x, y, w, h) in faces:

        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Afficher le cadre résultant

    cv2.imshow('Video', frames)


    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

video_capture.release()

cv2.destroyAllWindows()

# control+c dans la console pour fermer le programme 
#MasterLipakumu