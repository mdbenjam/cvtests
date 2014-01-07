import cv
import cv2

cv.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
camera_index = 0
capture = cv.CaptureFromCAM(camera_index)
c = cv.WaitKey(5000)
def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    frame = cv.QueryFrame(capture)

    if frame:
        gray_image = cv2.cvtColor(frame.channels, cv2.COLOR_BGR2GRAY)
        #print dir(frame)
        #har = cv2.cornerHarris(frame, 10, 2, .05)
        cv.ShowImage("w1", gray_image)
        #cv.ShowImage("w1", har)
    c = cv.WaitKey(10)
while True:
    repeat()

