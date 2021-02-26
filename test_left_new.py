#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_left.py    
@Contact :   xiaorun@nebula-link.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/25 0025 21:13   gxrao      1.0         None
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/25 17:21
# @Author  : xiaorun
# @Site    :
# @File    : mian.py
# @Software: PyCharm
import cv2
import numpy as np
from stereo import uv2XYZ
from new_version import findball
from shapely.geometry import Polygon
import time
#设置足球得阈值
lower_yello=np.array([4,94,93])
upper_yello=np.array([21,240,200])

capLeft=cv2.VideoCapture("left.avi")
capRight=cv2.VideoCapture("right.avi")
left=(505,75),(891,68),(1262,659),(1220,1002),(88,970)
roi_left=Polygon(left)

right=(490,94),(861,77),(1049,1012),(29,989),(134,344)
roi_right=Polygon(right)
#roi.contains()
####判断左右摄像头是否打开################
retLeft,frameLeft=capLeft.read()
print(frameLeft.shape)
retRight,frameRght=capRight.read()
if (not retLeft) or (not retRight):
    print("camera erro,can't open...")
findFootball_left=findball(lower_yello,upper_yello,roi_left)
findFootball_right=findball(lower_yello,upper_yello,roi_right)
while True:
    time_=time.time()
    retLeft,frameLeft=capLeft.read()
    retRight, frameRight = capRight.read()

    for i in range(len(left)-1):
        cv2.line(frameLeft,left[i],left[i+1],(0,0,255),3)
    cv2.line(frameLeft, left[4], left[0], (0, 0, 255),3)

    for j in range(len(right)-1):
        cv2.line(frameRight,right[j],right[j+1],(0,0,255),3)
    cv2.line(frameRight,right[4],right[0],(0,0,255),3)


    frame_left=findFootball_left.imageProcess(frameLeft)

    frame_right=findFootball_right.imageProcess(frameRight)

    if (frame_right ) and (frame_left ):
        x_left=int(frame_left[0])
        y_left=int(frame_left[1])
        w_left=int(frame_left[2])
        h_left=int(frame_left[3])
        cv2.rectangle(frameLeft, (x_left, y_left), (x_left + w_left, y_left + h_left), (255, 0, 0), 3)


        x_right,y_right,w_right,h_right=frame_right[0],frame_right[1],frame_right[2],frame_right[3]
        cv2.rectangle(frameRight, (x_right, y_right), (x_right + w_right, y_right + h_right), (255, 0, 0), 3)


        u_left,v_left=x_left+w_left//2,y_left+h_left//2
        u_right,v_right=x_right+w_right,y_right+h_right//2
        uv_left=(u_left,v_left)
        uv_right=(u_right,v_right)
        world3pt=uv2XYZ(uv_left,uv_right)


        if world3pt[2]<0:
            world3pt[2]=0
        XX,YY,ZZ=int(world3pt[0]),int(world3pt[1]),int(world3pt[2])
        label="x:{} y:{} z:{}".format(XX,YY,ZZ)
        cv2.putText(frameLeft,label,(30,30), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)
        frameLeft = cv2.resize(frameLeft, (640, 512))
        cv2.imshow("leftProcess", frameLeft)
        frameRight = cv2.resize(frameRight, (640, 512))
        cv2.imshow("rightProcess", frameRight)
    #frame_right=findFootball_left.imageProcess(frameRight)
    #frame_right= cv2.resize(frame_right, (frame_right.shape[1] // 2, frame_right.shape[0] // 2))
    #cv2.imshow("rightProcess", frame_right)
    print(time.time()-time_)
    if cv2.waitKey(50)==ord("q"):
        break

capLeft.release()
capRight.read()
