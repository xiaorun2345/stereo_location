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
from findFootball import findball
#设置足球得阈值
lower_yello=np.array([7,10,90])
upper_yello=np.array([15,255,120])

capLeft=cv2.VideoCapture("left.avi")
capRight=cv2.VideoCapture("right.avi")
####判断左右摄像头是否打开################
retLeft,frameLeft=capLeft.read()
retRight,frameRght=capRight.read()
if (not retLeft) or (not retRight):
    print("camera erro,can't open...")

findFootball=findball(lower_yello,upper_yello)
while True:
    retLeft,frameLeft=capLeft.read()
    retRight,frameRght=capRight.read()
    cv2.imshow("left",frameLeft)
    #cv2.imshow("right",frameRght)
    frame=findFootball.imageProcess(frameLeft)
    cv2.imshow("leftProcess",frame)


    if cv2.waitKey(10)==ord("q"):
        break

capLeft.release()
capRight.read()