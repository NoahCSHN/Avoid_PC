#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:53:09 2021

@author: bynav
"""

# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def get_new_size(img, scale):
    if type(img) != cv2.UMat:
        return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    else:
        return tuple(map(int, np.array(img.get().shape[:2][::-1]) * scale))

def get_max_scale(img, max_w, max_h):
    if type(img) == cv2.UMat:
        h, w = img.get().shape[:2]
    else:
        h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale

class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = get_max_scale(img, max_w, max_h)
        self._new_size = get_new_size(img, self.scale)
        self.__new_img = None

    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size, interpolation=cv2.INTER_AREA)
        return self.__new_img

def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    a = AutoScale(img, *new_wh)
    new_img = a.new_img
    if type(new_img) == cv2.UMat:
        h, w = new_img.get().shape[:2]
    else:
        h, w = new_img.shape[:2]
    padding = [(new_wh[1] - h)-int((new_wh[1] - h)/2), int((new_wh[1] - h)/2), (new_wh[0] - w)-int((new_wh[0] - w)/2), int((new_wh[0] - w)/2)]
    # new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - h, 0, new_wh[0] - w, cv2.BORDER_CONSTANT, value=color)
    new_img = cv2.copyMakeBorder(new_img, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=color)
    # logging.debug(f'image padding: {padding}') #cp3.6
    logging.debug('image padding: %s',padding) #cp3.5
    return new_img, (new_wh[0] / a.scale, new_wh[1] / a.scale), padding


def StereoVideo2pic(Vsource,Ipath):
    Vdataset = LoadStereoImages(Vsource)
    file_dir = str(Path(Ipath).absolute())
    left_file_dir = os.path.join(file_dir,'left_416')
    right_file_dir = os.path.join(file_dir,'right_416')
    if not os.path.isdir(left_file_dir):
        os.makedirs(left_file_dir)
    if not os.path.isdir(right_file_dir):
        os.makedirs(right_file_dir)
    i = 0
    for path, image_left, image_right, im0s, vid_cap in Vdataset:
        imgl_path=os.path.join(left_file_dir, str(i)+'_left.png')
        imgr_path=os.path.join(right_file_dir, str(i)+'_right.png')
        cv2.imwrite(imgl_path, image_left)
        cv2.imwrite(imgr_path,image_right)
        i += 1

class LoadStereoImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='\n')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ')

        # Padded resize
        h = img0.shape[0]
        w = img0.shape[1]
        w1 = round(w/2)
        img0_left = img0[:,:w1,:]
        img0_right = img0[:,w1:,:]
        img0_left = letterbox(img0_left,(416,416))[0]
        img0_right = letterbox(img0_right,(416,416))[0]


        return path, img0_left, img0_right, img0_left.shape, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
    
if __name__ == '__main__':
    StereoVideo2pic(Vsource = '/home/bynav/AI_SGBM/data/images/yyc/stereo_test/calib', Ipath = '/home/bynav/AI_SGBM/data/images/yyc/stereo_test')
