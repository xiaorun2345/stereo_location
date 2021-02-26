#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   findFootball.py    
@Contact :   xiaorun@nebula-link.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/25 0025 22:19   gxrao      1.0         None
'''
import cv2
import numpy as np
import math
from shapely.geometry import Point
#图像识别足球，对外接口，设置ROI区域，设置视频参数
#内部接口，
class findball():
    def __init__(self,lower,higher,roi):
        self.lower, self.higher=lower,higher
        self.roi=roi

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
        lower_yellos,upper_yellos = self.setValue()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # get mask
        mask = cv2.inRange(hsv, lower_yellos, upper_yellos)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # 灰度处理

        ret, binary = cv2.threshold(res, 20, 255, cv2.THRESH_BINARY)
        binary = cv2.erode(binary, None, iterations=1)
        binary=cv2.dilate(binary,None,iterations=2)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)<1:
            return []
        for con in contours:
            x, y, w, h = cv2.boundingRect(con)  # 将轮廓分解为识别对象的左上角坐标和宽、高
            if self.roi.contains(Point(x,y)):

                area=cv2.contourArea(con)
                if (50<area<60*60*4) :
                    # 在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    return [x,y,w,h]

            else:
                return []