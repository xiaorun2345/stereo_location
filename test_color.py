# import the necessary packages
from collections import deque
import numpy as np
import cv2
import time

# define the lower and upper boundaries of the "yellow"
# ball in the HSV color space, then initialize the list of tracked points
yellowLower = (7,10,90)
yellowUpper = (15,255,250)
pts = deque(maxlen=20)

cap = cv2.VideoCapture("left.avi")

# allow the camera or video file to warm up
max_area=0
# keep looping
while True:
    # grab the current frame
    ret, frame1 = cap.read()
    print(frame1.shape)
    frame=frame1[300:1000,100:1000]
    # blur it and convert it to the HSV color space
    #blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    #mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        temp=0
        for i in range(len(cnts)):
            area = cv2.contourArea(cnts[i])
            if area>max_area:
                cnt = cnts[i]
                max_area = area
                temp=i
        print(temp,i)
        #c = max(cnts, key=cv2.contourArea)
        c=cnts[temp]
        print(c)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame1, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame1, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
        cv2.line(frame1, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(50) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()