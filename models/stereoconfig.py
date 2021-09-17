#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:08:03 2021

@author: bynav
"""

import cv2
import numpy as np
from utils.general import calib_type
 
####################仅仅是一个示例###################################
 
 
# 双目相机参数
class stereoCamera(object):
    def __init__(self,mode=1,height=1280,width=960):
        """
        @description  : get rectification and depth calculation parameters
        ---------
        @param  : mode, class callb_type, choose camera run mode
        @param  : height, image height for rectification
        @param  : width, image width for rectification
        -------
        @Returns  : the Matrix Q
        -------
        """
        if mode==calib_type.OV9714_1280_720: 
            # %% OV9714 1280x720
            # 左相机内参
            print('Camera OV9714 1280X720',end='--')
            print('Out of time')
            self.width = 1280
            self.height = 720
            self.cam_matrix_left = np.array([[1147.039625617031, 0, 733.098961811485],
                                            [0., 1150.152056805138, 380.840107187423],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[1143.705662745462, 0, 708.710387078841],
                                            [0., 1145.008623150225, 378.384254182898],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[-0.008570670886844, 0.205372331430449, 0.000333267442408, 0.013776316839033, -0.568892690808155]])
            self.distortion_r = np.array([[0.017680717249695, 0.063462410840028, -0.000293093935331, 0.013855048079951, -0.189246079109404]])
    
            # 旋转矩阵
            self.R = np.array([[0.999999732264347, -0.000596922024167, 0.000423267446915],
                            [0.000597153117432,  0.999999672614051, -0.000546058554318],
                            [-0.000422941353966, 0.000546311163595, 0.999999761332333]])
    
            # 平移矩阵
            self.T = np.array([[-59.923725127564538], [0.009422096967625], [-2.535572114734797]])
    
            # 焦距 unit:pixel resolution ratio ，1280*720 1191.47 640*640 595.735 416*416 387.23   3.6mm
            self.focal_length = 387.23

            # 焦距 unit:pixel 949.62
            self.focal_length_pix = 1191.47  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 59.923725127564538  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00300
        elif mode==calib_type.AR0135_1280_720.value:
            # %% AR0135 1280x720
            # 左相机内参
            print('Camera AR0135 1280x720',end='--')
            print('Out of time')
            self.width = 1280
            self.height = 720
            self.cam_matrix_left = np.array([[899.9306, 0, 672.5951],
                                            [0., 903.0798, 472.7901],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[901.7015, 0, 670.4376],
                                            [0., 904.3837, 495.7119],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.0124, 0.0010, 0.0000, 0.0064, -0.0902]])
            self.distortion_r = np.array([[0.0112, -0.0053, -0.0006, 0.0070, -0.0457]])
    
            # 旋转矩阵
            self.R = np.array([[0.999999544012909, -0.000361332677693, -0.000883975491791],
                            [0.000358586486588, 0.999995115724826, -0.003104825634915],
                            [0.000885093049172, 0.003104507237489, 0.999994789308978]])
    
            # 平移矩阵
            self.T = np.array([[-44.789005274626312], [0.369693318112806], [-1.106236740447207]])
    
            # 焦距 unit:pixel resolution ratio ，1280*720 932.83 640*640 466.415 416*416 303.17
            self.focal_length = 303.17

            # 焦距 unit:pixel 949.62
            self.focal_length_pix = 932.83  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 44.789005274626312  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375
        elif mode == calib_type.AR0135_1280_960.value:
            # %% 1280x960
            # 左相机内参
            print('Camera AR0135 1280x960',end='--')
            print('Up to date')
            self.width = 1280
            self.height = 960
            self.cam_matrix_left = np.array([[923.642308115181, 0, 661.563612365323],
                                            [0., 927.277662980802, 473.585058718788],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[920.534932214038, 0, 650.83405580391],
                                            [0., 922.887792853376, 505.013813754372],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.0205144610251091, 0.00162711308435883, 0.00309271639752063, 0.003644684681153, -0.0711454543707475]])
            self.distortion_r = np.array([[0.0131449540956766, -0.00937955020237468, 0.00343709188166571, 0.00193102960455661, 0.0162388757315773]])
    
            # 旋转矩阵
            self.R = np.array([[0.999928614176774, 0.00148341625281357, -0.0118560544337673],
                            [-0.00140644636358242, 0.999977900426304, 0.00649773557688742],
                            [0.0118654312665798, -0.00648059682603934, 0.999908602526069]])
    
            # 平移矩阵
            self.T = np.array([[-45.1754833198736], [0.327319805179798], [-1.00710987782975]])
    
            # 焦距 unit:pixel resolution ratio ，1280*960 928.778 640*640 464.389 416*416 344.906
            # if width == 1280:
            # self.focal_length = 1068.62989
            # elif width == 640:
            #     self.focal_length = 534.314945
            # else:
            self.focal_length = 996.44

            # 焦距 unit:pixel 928.778
            self.focal_length_pix = 996.44  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 45.1754833198736  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375
            
        elif mode == calib_type.AR0135_416_416.value:
            # %% 416x416
            # 左相机内参
            print('Camera AR0135 416x416',end='--')
            print('Up to date')
            self.width = 416
            self.height = 416
            self.cam_matrix_left = np.array([[299.84033260309, 0, 215.430182780566],
                                            [0., 301.082445397163, 206.370313144114],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[299.070221795856, 0, 212.197748631633],
                                            [0., 299.87300452586, 216.551097695313],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.0287950022612231, -0.0375079708523528, 0.0033644344968114, 0.00356954375215185, -0.0181762649399843]])
            self.distortion_r = np.array([[0.022963587576208, -0.052296039256012, 0.00363477481406291, 0.00214578461198214, 0.0705387035339141]])

            # 平移矩阵
            self.T = np.array([[-45.2826966723642],[0.338492907296872],[-0.700828312824103]])
        
            # 旋转矩阵
            self.R = np.array([[0.999935799111265, 0.001546582675125, -0.0112252277368931],
                            [-0.00147466919479294, 0.999978356413656, 0.00641186829312914],
                            [0.0112349012671246, -0.00639490314793815, 0.999916437612287]])

            # 焦距 unit:pixel resolution ratio 416*416 349.124
            self.focal_length = 324.1467

            # 焦距 unit:pixel 349.124
            self.focal_length_pix = 324.1467  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 45.2826966723642  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375     

        elif mode == calib_type.AR0135_640_640.value:
            # %% 640x640
            # 左相机内参
            print('Camera AR0135 640x640',end='--')
            print('Up to date')
            self.width = 640
            self.height = 640
            self.cam_matrix_left = np.array([[461.844833948282, 0, 331.403192583902],
                                            [0., 463.711296692977, 316.784810501856],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[460.246022343269, 0, 326.067704392229],
                                            [0., 461.447294239966, 332.546977631754],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.0224101980393075, -0.00802132393404491, 0.00296299386878469, 0.00385699064967932, -0.0575143921289213]])
            self.distortion_r = np.array([[0.0163278657370845, -0.0226085245961337, 0.00328929552458683, 0.00219139151122059, 0.0308441411744662]])
    
            # 旋转矩阵
            self.R = np.array([[0.999928811085943, 0.00146532069116139, -0.0118416888797451],
                            [-0.00138753491040948, 0.999977426039445, 0.00657435612052772],
                            [0.0118510551059819, -0.00655745734253603, 0.999908271916017]])  
                              
            # 平移矩阵
            self.T = np.array([[-45.2018551679078], [0.323914220091648], [-1.05093440368482]])
    
            # 焦距 unit:pixel resolution ratio 640x640 349.124
            self.focal_length = 495.84

            # 焦距 unit:pixel 349.124
            self.focal_length_pix = 495.84  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 45.2018551679078  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375    
            
        elif mode == calib_type.AR0135_640_480.value:
            # %% 640x640
            # 左相机内参
            print('Camera AR0135 640x480',end='--')
            print('Out of time')
            self.width = 640
            self.height = 480
            self.cam_matrix_left = np.array([[464.439312296583, 0, 342.548068699737],
                                            [0., 465.457738211478, 233.651433626778],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[462.170632901995, 0, 338.079646689704],
                                            [0., 462.26842453768, 245.78819336519],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.0219089239325666, 0.00460092573614185, -0.00646591760327415, 0.0109270275959497, -0.0867620227422152]])
            self.distortion_r = np.array([[0.0272662496202452, -0.0424029960645065, -0.00484492636909076, 0.0100349479349904, 0.00826485872616951]])
    
            # 旋转矩阵
            self.R = np.array([[0.999975518009243, 0.000152432767973579, -0.00699572343628832],
                            [-0.000126983617991523, 0.999993373974003, 0.00363811534319693],
                            [0.00699623165043493, -0.00363713793261832, 0.999968911501929]])  
                              
            # 平移矩阵
            self.T = np.array([[-45.0518881011289], [0.414329549832418], [-1.42608145944905]])
    
            # 焦距 unit:pixel resolution ratio 640*480 403.691 416x416 318.054100095
            self.focal_length = 318.054100095

            # 焦距 unit:pixel 349.124
            self.focal_length_pix = 489.314  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 45.0729106116333  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375   

        else:                                
            # %% 416X416
            # 左相机内参
            print('Datasets Middlebury 416x416',end='--')
            print('Updated')
            self.width = 416
            self.height = 416
            self.cam_matrix_left = np.array([[4152.073, 0, 1288.147],
                                            [0., 4152.073, 973.571],
                                            [0., 0., 1.]])
            # 右相机内参
            self.cam_matrix_right = np.array([[4152.073, 0, 1501.231],
                                            [0., 4152.073, 973.571],
                                            [0., 0., 1.]])
    
            # 左右相机畸变系数:[k1, k2, p1, p2, k3]
            self.distortion_l = np.array([[0.0219089239325666, 0.00460092573614185, -0.00646591760327415, 0.0109270275959497, -0.0867620227422152]])
            self.distortion_r = np.array([[0.0272662496202452, -0.0424029960645065, -0.00484492636909076, 0.0100349479349904, 0.00826485872616951]])
    
            # 旋转矩阵
            self.R = np.array([[0.999975518009243, 0.000152432767973579, -0.00699572343628832],
                            [-0.000126983617991523, 0.999993373974003, 0.00363811534319693],
                            [0.00699623165043493, -0.00363713793261832, 0.999968911501929]])  
                              
            # 平移矩阵
            self.T = np.array([[-45.0518881011289], [0.414329549832418], [-1.42608145944905]])
    
            # 焦距 unit:pixel resolution ratio 640*480 403.691 416x416 318.054100095
            self.focal_length = 318.054100095

            # 焦距 unit:pixel 349.124
            self.focal_length_pix = 489.314  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

            # 基线距离
            self.baseline = 45.0729106116333  # 单位：mm， 为平移向量的第一个参数（取绝对值）
            
            # pixel size unit: mm
            self.pixel_size = 0.00375   

        # 计算校正变换
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(self.cam_matrix_left, self.distortion_l, self.cam_matrix_right,\
                                                                                             self.distortion_r, (self.width, self.height), self.R, self.T, alpha=0)
    
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cam_matrix_left, self.distortion_l, self.R1, self.P1, (self.width, self.height), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cam_matrix_right, self.distortion_r, self.R2, self.P2, (self.width, self.height), cv2.CV_32FC1)
        print(self.Q)
