'''
Author: your name
Date: 2021-09-17 10:00:44
LastEditTime: 2021-09-17 10:38:17
LastEditors: Please set LastEditors
Description: get focal length parameter
FilePath: /AI_SGBM/cam_calib.py
'''

from models.stereoconfig import stereoCamera
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_num",type=int,default=1,help="the camera calibration resolution ratio")
    opt = parser.parse_args()
    print(opt)
    cam_mode = stereoCamera(opt.cam_num)
    print(cam_mode)
