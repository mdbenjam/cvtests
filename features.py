import cv
import cv2
import numpy as np

cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
capture = cv2.VideoCapture(0)

gx = gy = 1
grayscale = blur = canny = False
grayscale = True
features = True

p1 = 1000
p2 = .001
k = 2
lm = 1
lmo = np.ones([lm,lm])

def local_max(image):
    global lmo
    a = image > cv2.dilate(image, lmo)
    print a
    return a

def repeat():
    global capture #declare as globals since we are assigning to them now
    global gx, gy, grayscale, canny, blur, features
    global p1, p2, k, lm
    ret, frame = capture.read()
    orig = frame
    c = cv.WaitKey(10)
    if frame is None:
        return

    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if blur:
        frame = cv2.GaussianBlur(frame, (11,11), gx, gy)

    if grayscale and canny:
        frame = cv2.Canny(frame, 10, 100, 3)

    elif grayscale and features:
        corners = cv2.goodFeaturesToTrack(frame, p1, p2, k)
        if corners is not None:

            frame = orig
            print len(corners)
            for corner in corners:
                x = corner[0,0]
                y = corner[0,1]
                print x,y
                frame[y,x,2] = 255
        else:
            print ':D'

        # img2 = np.zeros_like(orig)
        # img2[:,:,0] = orig[:,:,0]
        # img2[:,:,1] = orig[:,:,1]
        # img2[:,:,2] = orig[:,:,2]/2 + corner_image / 2
        # frame = img2

    if c==ord('='):
        gx += 2
        gy += 2
    elif c == ord('-'):
        gx = max(1, gx-2)
        gy = max(1, gy-2)
    elif c == ord('x'):
        gx += 2
    elif c == ord('X'):
        gx = max(1, gx-2)
    elif c == ord('q'):
        exit(0)

    elif c == ord('b'):
        blur = not blur
    elif c == ord('g'):
        grayscale = not grayscale
    elif c == ord('c'):
        canny = not canny
    elif c == ord('h'):
        features = not features

    elif c == ord('1'):
        p1 += 1
    elif c == ord('`'):
        p1 -= 1
    elif c == ord('k'):
        k += .05
    elif c == ord('l'):
        k -= .05
    elif c == ord('2'):
        p2 += 2
    elif c == ord('3'):
        p2 -= 2

    elif c == ord('6'):
        lm -= 2
        lm = max(1,lm)
        lmo = np.ones([lm,lm])
    elif c == ord('7'):
        lm += 2
        lmo = np.ones([lm,lm])

    p1 = max(1, p1)
    p2 = max(1, min(31, p2))

    cv2.putText(
        frame,
        "p1: %d, p2: %d, k: %f, lm: %d" % (p1, p2, k, lm),
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        0
    )

    cv2.imshow("w1", frame)

while True:
    repeat()

