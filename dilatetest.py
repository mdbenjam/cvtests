import numpy as np
import cv2
import cv


cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)

m = cv2.imread('images/mantis.jpg')
lm = 1
while True:
    m2 = cv2.dilate(m, np.ones([lm,lm]))
    print m >= m2

    c = cv.WaitKey(10)

    if c == ord('-'):
        lm -= 2
    if c == ord('='):
        lm += 2
    lm = max(1, lm)

    cv2.putText(
        m2,
        "lm: %d" % (lm),
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        0xffffff
    )

    cv2.imshow('w1', m2)

    if c == ord('q'):
        break
