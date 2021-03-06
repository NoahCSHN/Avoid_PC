'''
Author: Noah
Date: 2021-04-29 20:02:24
LastEditTime: 2021-09-18 08:35:58
LastEditors: Please set LastEditors
Description: image rectification and resize
FilePath: /AI_SGBM/utils/img_preprocess.py
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:06:14 2021

@author: bynav
"""

# -*- coding: utf-8 -*-
import cv2,time,logging
import numpy as np
from models.stereoconfig import stereoCamera
import os
from pathlib import Path
from utils.datasets import letterbox
from utils.general import timethis,timeblock
# from pcl import pcl_visualization
 

# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if(img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
 
    return img1, img2

 
# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
 
    return undistortion_image
 
 
# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
# @timethis
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T
 
    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0)
 
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
 
    return map1x, map1y, map2x, map2y, Q
 
 
# 畸变校正和立体校正
# @timethis
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
 
    return rectifyed_img1, rectifyed_img2
 
 
# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    return output
 
 
# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }
 
    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)
 
    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
 
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]
 
        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right
 
    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.
 
    return trueDisp_left, trueDisp_right
 
 
# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]
 
    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)
 
    points_ = np.hstack((points_1, points_2, points_3))
 
    return points_
 
 
# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols
 
    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)
 
    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)
 
    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)
 
    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)
 
    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]
 
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack((remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))
 
    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)
 
    return pointcloud_1

# @timethis
def resize_convert(imgl_rectified, imgr_rectified, imgsz=640, stride=32):
    imgl_rectified = letterbox(imgl_rectified, imgsz, stride=stride)[0]
    imgr_rectified = letterbox(imgr_rectified, imgsz, stride=stride)[0]
    
    # Convert
    img_ai = imgl_rectified[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img_ai = np.ascontiguousarray(img_ai)
    imgl_rectified = np.ascontiguousarray(imgl_rectified)
    imgr_rectified = np.ascontiguousarray(imgr_rectified)
    return img_ai, imgl_rectified, imgr_rectified
 
# @timethis
def Image_Rectification(camera_config, img_left, img_right, im0sz=(1280,720), imgsz=640, stride=32, path=False, UMat=False, debug=False):
    '''
    description: image rectification and resize
    param {camera_config: class, the camera configuration parameters
           img_left: matrix or str, image from left camera
           img_right: matrix or str, image from right camera
           im0sz: int, the original image size
           imgsz: int, the image size after resize
           stride: int, the ratio of downsample
           path: bool: if true, the imput of img_left and img_right are str of path;otherwise, they are photo
           UMat: bool: if true, operate all image on gpu 
           debug: bool: if true, save the rectified image
    }
    return {iml_rectified: the rectified and resized image of the left camera
            imr_rectified: the rectified and resized image of the right camera
            img_ai: the same as iml_rectified 
            img_ai_raw: the rectified but not resized image of the left camera
    }
    '''
    # 读取MiddleBurry数据集的图片
    t0 = time.time()
    # with timeblock('read file'):
    if path:
        imgl_path=str(Path(img_left).absolute())
        imgr_path=str(Path(img_right).absolute())
        iml = cv2.imread(imgl_path)  # left
        imr = cv2.imread(imgr_path)  # right
    else:
        iml = img_left  # left
        imr = img_right # right           
    # 读取相机内参和外参
    config = camera_config
    img_ai_raw = iml
    # img_ai_raw = img_ai_raw[:,:,::-1].transpose(2,0,1)
    # img_ai_raw = np.ascontiguousarray(img_ai_raw)
    # 图像缩放
    if config.width != 1280:
        img_ai, iml, imr = resize_convert(iml, imr, imgsz, stride)
    else:
        img_ai = letterbox(iml,new_shape=(imgsz,imgsz),stride=stride)[0]
        img_ai = img_ai[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ai = np.ascontiguousarray(img_ai)
    if UMat:
        iml = cv2.UMat(iml)
        imr = cv2.UMat(imr)
        # 立体校正
        # map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = rectifyImage(iml, imr, config.map1x, config.map1y, config.map2x, config.map2y)
        iml_rectified = cv2.UMat.get(iml_rectified)
        imr_rectified = cv2.UMat.get(imr_rectified)
    else:
        # 立体校正
        # map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = rectifyImage(iml, imr, config.map1x, config.map1y, config.map2x, config.map2y)
 
    if debug:
    # 绘制等间距平行线，检查立体校正的效果
        cv2.imwrite('/home/bynav/0_code/AI_SGBM/runs/detect/exp/Left1_rectified.bmp', iml_rectified)
        cv2.imwrite('/home/bynav/0_code/AI_SGBM/runs/detect/exp/Right1_rectified.bmp', imr_rectified)  
        # print(Q)           
        line = draw_line(iml_rectified, imr_rectified)
        cv2.imwrite('/home/bynav/0_code/AI_SGBM/runs/detect/exp/line.png', line)
 
    # 立体匹配
    # iml_rectified, imr_rectified = preprocess(iml_rectified, imr_rectified)  # 预处理，一般可以削弱光照不均的影响，不做也可以
    # disp, _ = stereoMatchSGBM(iml_, imr_, True)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
    # cv2.imwrite('/home/bynav/AI_SGBM/data/images/yyc/stereo_test/视差.png', disp)
 
    # 计算像素点的3D坐标（左相机坐标系下）
    # points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数
 
    # 构建点云--Point_XYZRGBA格式
    # pointcloud = DepthColor2Cloud(points_3d, iml)
 
    # 显示点云
    # view_cloud(pointcloud()
    logging.info(f'Image rectification Done. ({time.time() - t0:.3f}s)')
    return iml_rectified,imr_rectified,img_ai,img_ai_raw

if __name__ == '__main__':
    config = stereoCamera(3)
    img_left = '../data/images/Left1.bmp'
    img_right = '../data/images/Right1.bmp'
    left,right,left_rgb = Image_Rectification(config, img_left, img_right, path=True)
    cv2.imshow('left',left)
    cv2.waitKey(500)
    cv2.imshow('right',right)
    cv2.waitKey(500)
    cv2.imshow('left_rgb',left_rgb)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
