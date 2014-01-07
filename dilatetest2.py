import numpy as np
import cv2
import cv


cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("w2", cv.CV_WINDOW_AUTOSIZE)

m = cv2.imread('images/mantis.jpg')
cv2.imshow('w1', m)


def local_max(image):
    lm = 5
    ones = np.ones([lm,lm])
    # ones[lm/2,lm/2] = 1
    a = image >= cv2.dilate(image, ones)
    return a * 255

m2 = local_max(m)

cv2.imshow('w1', m)
cv2.imshow('w2', m2)

raw_input()
