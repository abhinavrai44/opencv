import cv2
import sys
import cv2 as cv
import numpy as np
from common import clock, draw_str


# Create the haar cascade
faceCascade = cv2.CascadeClassifier("/home/abhinav/opencv-2.4.9/data/lbpcascades/lbpcascade_frontalface.xml")

# Read the video
cv2.namedWindow('image')
cap = cv2.VideoCapture(0)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while (True):
    ret, img  = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    t = clock()
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    #print "Found {0} faces!".format(len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    dt = clock() - t
    draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))
    
    cv2.imshow('image',img)
    if 0xFF & cv2.waitKey(5) == 27:
            break

cv2.destroyAllWindows()
