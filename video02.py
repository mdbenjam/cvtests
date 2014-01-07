import cv

cv.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
camera_index = 0
capture = cv.CaptureFromCAM(camera_index)
c = cv.WaitKey(5000)
def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    frame = cv.QueryFrame(capture)

    if frame:

        cv.ShowImage("w1", frame)
    c = cv.WaitKey(10)
while True:
    repeat()

