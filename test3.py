import cv2
import numpy as np
c = cv2.VideoCapture(0)

while True:
    _,f = c.read()
    if f:
        cv2.imshow('e2',f)
    if cv2.waitKey(10)==27:
        break

cv2.destroyAllWindows()

