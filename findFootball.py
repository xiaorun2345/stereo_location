#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/25 16:39
# @Author  : xiaorun
# @Site    : 
# @File    : findFootball.py
# @Software: PyCharm
import cv2
import numpy as np

#图像识别足球，对外接口，设置ROI区域，设置视频参数
#内部接口，
class findball():
    def __init__(self,lower,higher):
        self.lower, self.higher=lower,higher

    def setValue(self):
         self.lower=np.array(self.lower)
         self.higer=np.array(self.higher)
         return self.lower,self.higher

    # def isArea(self,x,y,roi):
    #     point = Point([x, y])
    #     polygon = Polygon(roi)
    #     return polygon.contains(point)
    #
    #     return 0
    def imageProcess(self,frame):
        # 内核
        kernel = np.ones((5, 5), np.uint8)
        lower_yellos,upper_yellos = self.setValue()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # get mask
        mask = cv2.inRange(hsv, lower_yellos, upper_yellos)
        cv2.imshow("mask",mask)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        #mask = cv2.dilate(mask,None, iterations=1)
        mask=cv2.erode(mask,kernel,iterations=15)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask= cv2.GaussianBlur(closing, (5, 5), 0)
        cv2.imshow("erod", mask)
        ret, binary = cv2.threshold(mask, 12, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for con in contours:
            x, y, w, h = cv2.boundingRect(con)  # 将轮廓分解为识别对象的左上角坐标和宽、高
            # 在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        return frame