#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/19 10:38
# @Author  : xiaorun
# @Site    : 
# @File    : stereo.py
# @Software: PyCharm
import cv2
import numpy as np

#####################左右相机外参#################################################
###########左相机世界坐标与图像坐标
object_3d_points_L = np.array(([-1300, 1200, 0],
                            [1400, 1360, 0],
                            [500, 2100,  0],
                            [-700, 2400, 0],
                            [-300,3000,0],
                            [1300,3100,0],
                            [-800,3350,0],
                            [-1200,4260,0],
                            [100,4650,0],
                            [1350,4920,0]),dtype=np.double)

object_2d_point_L= np.array(([535, 662],
                            [1038, 594],
                            [839, 494],
                            [641, 471],
                            [694,403],
                            [909,383],
                            [619,373],
                            [567,304],
                            [718,273],
                            [847,250]),dtype=np.double)
##########左相机世界图像坐标


###########右相机世界坐标与图像坐标
object_3d_points_R= np.array(([-1300, 1200, 0],
                            [1400, 1360, 0],
                            [500, 2100,  0],
                            [-700, 2400, 0],
                            [-300,3000,0],
                            [1300,3100,0],
                            [-800,3350,0],
                            [-1200,4260,0],
                            [100,4650,0],
                            [1350,4920,0]),dtype=np.double)

object_2d_point_R= np.array(([271, 619],
                            [792, 693],
                            [629, 504],
                            [449, 457],
                            [529, 397],
                            [758,400],
                            [470,361],
                            [451,289],
                            [612,270],
                            [756,259]),dtype=np.double)
##########右相机世界图像坐标
########################################################################


#################左右相机内参###############################################
camera_matrix_L = np.array(([548.2646, 0, 666.9979],
                         [0, 549.5398, 494.3196],
                         [0, 0, 1.0]), dtype=np.double)
dist_coefs_L = np.array([0.0119, -0.0018, 0, 0], dtype=np.double)

rotM_L=np.array(([ 0.99647804, -0.0835722,   0.00687046],
                 [-0.06215961, -0.7911815,  -0.60841435],
                 [ 0.05628231,  0.60584447, -0.79358981]),dtype=np.double)

tvec_L=np.array(([  734.59639027],
                 [ 1704.06440486],
                 [ 2086.22518369]),dtype=np.double)

RT_leftcamera=np.hstack((rotM_L,tvec_L))

m_left=np.matmul(camera_matrix_L ,RT_leftcamera)


camera_matrix_R=np.array(([554.9156,0,635.8487],
                          [0,555.5995,509.9433],
                          [0,0,1.0]),dtype=np.double)
dist_coefs_R=np.array([0.0256,-0.0133,0,0],dtype=np.double)

rotM_R=np.array(([ 0.99353492,  0.1123029,   0.01662595],
                 [ 0.10227696, -0.82186833, -0.56042115],
                 [-0.04927258,  0.55849843, -0.82804089]),dtype=np.double)

tvec_R=np.array(([ -773.49701673],
                 [ 1739.81170979],
                 [ 2156.35612044]),dtype=np.double)
RT_rightcamera=np.hstack((rotM_R,tvec_R))

#右相机M矩阵
#       [u1]      |X|					  [u2]      |X|
#     Z*[v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
#	    [ 1]      |Z|					  [ 1]      |Z|
#			      |1|								|1|

m_right=np.matmul(camera_matrix_R ,RT_rightcamera)
#################左右相机内参###############################################

####################通过图像坐标uv推导出世界坐标XYZ############################

def uv2XYZ(point2d_left,point2d_right):
    A=np.zeros((4,3))
    A[0,0]=point2d_left[0]*m_left[2,0]-m_left[0,0]
    A[0,1]=point2d_left[0]*m_left[2,1]-m_left[0,1]
    A[0,2]=point2d_left[0]*m_left[2,2]-m_left[0,2]

    A[1,0]=point2d_left[1]*m_left[2,0]-m_left[1,0]
    A[1,1]=point2d_left[1]*m_left[2,1]-m_left[1,1]
    A[1,2]=point2d_left[1]*m_left[2,2]-m_left[1,2]

    A[2,0]=point2d_right[0]*m_right[2,0]-m_right[0,0]
    A[2,1]=point2d_right[0]*m_right[2,1]-m_right[0,1]
    A[2,2]=point2d_right[0]*m_right[2,2]-m_right[0,2]

    A[3,0]=point2d_right[1]*m_right[2,0]-m_right[1,0]
    A[3,1]=point2d_right[1]*m_right[2,1]-m_right[1,1]
    A[3,2]=point2d_right[1]*m_right[2,2]-m_right[1,2]
    #最小二乘法B矩阵
    B=np.zeros((4,1))
    B[0,0]=m_left[0,3]-point2d_left[0]*m_left[2,3]
    B[1,0]=m_left[1,3]-point2d_left[1]*m_left[2,3]
    B[2,0]=m_right[0,3]-point2d_right[0]*m_right[2,3]
    B[3,0]=m_right[1,3]-point2d_right[1]*m_right[2,3]

    world_3d=np.matmul(np.matrix(A).I,B)
    #print(world_3d)

    xyz=cv2.solve(A,B,flags=cv2.DECOMP_SVD)

    return world_3d
####################通过图像坐标uv推导出世界坐标XYZ############################


####################通过世界坐标XYZ推导出图像坐标uv############################

#       [fx s x0]							[Xc]		[Xw]		[u]	  1		[Xc]
#	K = |0 fy y0|      TEMP = [R T]		    |Yc| = TEMP*|Yw|		| | =  —*K *|Yc|
#	    [ 0 0 1 ]							[Zc]		|Zw|		[v]	  Zc	[Zc]
#														[1 ]

def xyz2uv(worldPoint,intrinsic,translation,rotation):
    worldPoint=np.array((worldPoint),dtype=np.double)

    temp_value=np.matmul(np.hstack((rotation,translation)),worldPoint)

    uv=np.matmul(intrinsic,temp_value)
    return uv

left_point=(555,368)
right_point=(360,355)
point3=(-800,3350,0,1)

uv_left=xyz2uv(point3,camera_matrix_L,tvec_L,rotM_L)
uv_right=xyz2uv(point3,camera_matrix_R,tvec_R,rotM_R)
print(uv_left/uv_left[2])
print(uv_right/uv_right[2])
world_3d=uv2XYZ(left_point,right_point)
print(world_3d)
