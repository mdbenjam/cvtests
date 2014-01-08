import cv
import cv2
import math
import numpy as np
from sets import Set

cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)

img1 = cv2.imread('setcards.jpg')

width, height, depth = img1.shape
diag = np.sqrt(width**2 + height**2)

img2 = np.zeros(img1.shape)

frame = cv2.Canny(img1, 10, 100, 3)
# lines = cv2.HoughLines(frame, 1, math.pi/180, 200)

# COPIED:
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 404, 156, apertureSize=3)

lines = cv2.HoughLinesP(edges,1,np.pi/180,50, minLineLength=20, maxLineGap=2)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imshow("w1", edges)

var = {
    'threshold1': 6910,
    'threshold2': 2500,
    'apertureSize': 5,
    'houghParam1': 1,
    'houghParam2': np.pi/180,
    'houghParam3': 50,
    'minLineLength': 20,
    'maxLineGap': 2,
}
curvar = 'threshold1'
amt = 2.0
toShow = 'edges'

cv2.imshow("w1", img1)

disp_quad = 0

lasterr = None
while True:
    try:
        threshold1 = var['threshold1']
        threshold2 = var['threshold2']
        apertureSize = int(var['apertureSize'])
        edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize)

        houghParam1 = var['houghParam1']
        houghParam2 = var['houghParam2']
        houghParam3 = var['houghParam3']
        minLineLength = var['minLineLength']
        maxLineGap = var['maxLineGap']
        lines = cv2.HoughLines(edges, houghParam1, houghParam2, houghParam3*2, 0, 0) #minLineLength=minLineLength, maxLineGap=maxLineGap)
        intersections = []

        def make_segments(rho, theta):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + diag*(-b))
            y1 = int(y0 + diag*(a))
            x2 = int(x0 - diag*(-b))
            y2 = int(y0 - diag*(a))
            return (x1, y1, x2, y2)

        segments = []
        if lines is not None and lines[0] is not None:
            for rho, theta in lines[0]:
                segments.append(make_segments(rho, theta))


        intersections = []
        for i in range(len(segments)):
            intersections.append([])

        for i in range(len(segments)):
            these_intersections = []
            x1_i, y1_i, x2_i, y2_i = segments[i]
            p = np.array([x1_i, y1_i])
            r = np.array([x2_i - x1_i, y2_i - y1_i])
            for j in range(i, len(segments)):
                x1_j, y1_j, x2_j, y2_j = segments[j]
                q = np.array([x1_j, y1_j])
                s = np.array([x2_j - x1_j, y2_j - y1_j])
                denom = float(np.cross(r, s))
                if abs(denom) > .001:
                    v = q - p
                    t = np.cross(v,s)/denom
                    u = np.cross(v,r)/denom
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        x, y = p+t*r
                        x2, y2 = q+u*s
                        if abs(np.dot(r,s)/np.linalg.norm(r)/np.linalg.norm(s)) < .3:
                        #if 0 <= x <= width and 0 <= y <= height:
                            these_intersections.append([x,y,i,j])
                            intersections[j].append([x,y,i,j])
            intersections[i].extend(these_intersections)

        for i in range(len(intersections)):
            intersections[i] = sorted(intersections[i], key = lambda x: x[1])
            intersections[i] = sorted(intersections[i], key = lambda x: x[0])

        quads = Set([])

        def find_intersection_index(intersection, i):
            for k in range(len(intersection)):
                if (intersection[k][2] == i or
                    intersection[k][3] == i):
                    return k
            return -1

        def get_other_index(intersection, i):
            x,y,a,b = intersection
            other = a
            if other == i:
                other = b
            return other
             

        def make_quad(intersections):
            q = []
            for inter in intersections:
                q.append((int(inter[0]),int(inter[1])))
            return (q[0], q[1], q[2], q[3])

        search_range = 2

        for i in range(len(intersections)):
            for j in range(len(intersections[i])):
                other = get_other_index(intersections[i][j], i)
                
                index = find_intersection_index(intersections[other], i)

                for m in range(search_range):
                    m_sum = j+m+1
                    if m_sum < len(intersections[i]):
                        intersection_south = intersections[i][m_sum]
                        #print intersections[i][j], intersection_south
                        other_index_south = get_other_index(intersection_south, i)
                        #print i,other, other_index_south
                        for n in range(search_range):
                            n_sum = index+n+1
                            if n_sum < len(intersections[other]):
                                intersection_east = intersections[other][n_sum]
                                other_index_east = get_other_index(intersection_east, other)

                                #print intersections[other_index_east], other_index_south
                                found_index = find_intersection_index(intersections[other_index_east],other_index_south)
                                #print other,i,j,other_index_south, other_index_east
                                if found_index != -1:
                                    q = make_quad([intersections[i][j],
                                                            intersection_east,
                                                            intersections[other_index_east][found_index],
                                                            intersection_south])
                                    length = []
                                    flag = False
                                    for o in range(len(q)):
                                        p = (o + 1) % len(q)
                                        x1, y1 = q[o]
                                        x2, y2 = q[p]
                                        l = (x1-x2)**2+(y1-y2)**2
                                        length.append(l)
                                        if l <= 400:
                                            flag = True
                                    if not flag:
                                        for o in range(len(length)):
                                            p = (o + 1) % len(q)
                                            l1 = length[o]
                                            l2 = float(length[p])
                                            if not (.25 < l1/l2 < 4):
                                                flag = True
                                    
                                    if not flag:
                                        quads.add(q)


        print len(intersections)
        print len(quads)


        if toShow != 'edges':
            img1 = cv2.imread('setcards.jpg')
            img2 = np.zeros(img1.shape)

            index = 0
            for q in quads:
                if disp_quad == index:
                    arr = [np.array(q,'int32')]
                    print arr
                    cv2.fillPoly(img1,arr,(0,0,100))
                    cv2.fillPoly(img2,arr,(0,0,100))
                    break
                index = index + 1

            for x1, y1, x2, y2 in segments:
                cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

            for x,y,i,j in [val for subl in intersections for val in subl]:
                cv2.circle(img1, (int(x),int(y)), 3, (100, 100, 0), -1)
                cv2.circle(img2, (int(x),int(y)), 3, (100, 100, 0), -1)
                
        if toShow == 'edges':
            cv2.imshow("w1", edges)
        elif toShow == 'img1':
            cv2.imshow("w1", img1)
        elif toShow == 'img2':
            cv2.imshow("w1", img2)

        lasterr = None
    except Exception as e:
        if str(lasterr) != str(e):
            lasterr = e
            print e
        cv2.imshow("w1", np.zeros(img1.shape))

    c = cv.WaitKey(0)
    if c == ord('='):
        var[curvar] += amt
    elif c == ord('-'):
        if var[curvar] - amt > 0:
            var[curvar] -= amt
    elif c == ord('1'):
        curvar = 'threshold1'
    elif c == ord('2'):
        curvar = 'threshold2'
    elif c == ord('3'):
        curvar = 'apertureSize'
    elif c == ord('4'):
        curvar = 'houghParam1'
    elif c == ord('5'):
        curvar = 'houghParam2'
    elif c == ord('6'):
        curvar = 'houghParam3'
    elif c == ord('7'):
        curvar = 'minLineLength'
    elif c == ord('8'):
        curvar = 'maxLineGap'
    elif c == ord('['):
        amt /= 2
    elif c == ord(']'):
        amt *= 2
    elif c == ord('p'):
        print var,amt,toShow
    elif c == ord('z'):
        toShow = 'img1'
    elif c == ord('x'):
        toShow = 'img2'
    elif c == ord('c'):
        toShow = 'edges'
    elif c == ord('q'):
        exit(0)
    elif c == ord('n'):
        disp_quad = disp_quad + 1
    elif c == ord('m'):
        disp_quad = disp_quad - 1

raw_input()
