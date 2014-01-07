import numpy as np
import cv2
import cv


cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)

m = cv2.imread('images/cat.jpg')
cv2.imshow('w1', m)

rho = 1
theta = np.pi/180.
thresh = 30
frame = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
lines = cv2.HoughLinesP(frame, rho, theta, thresh)

print 'Lines computed', len(lines), len(lines[0])
for line in lines[0]:
    pt1 = (line[0], line[1])
    pt2 = (line[2], line[3])
    cv2.line(m,pt1,pt2, 0, 4)

print 'Done'
cv2.imshow('w1', m)

raw_input()
