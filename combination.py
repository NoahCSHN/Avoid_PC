#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:29:00 2021
@author: bynav
"""
import argparse,logging,time,os
import numpy as np
from utils.Stereo_Application import stereo_sgbm,SGBM,BM
from detect import YOLOv5
from utils.datasets import DataPipline, LoadStereoImages, StereoVideo2pic, LoadStereoWebcam
from utils.general import set_logging,check_requirements,scale_coords,find_cam,timethis
from utils.img_preprocess import Image_Rectification
from models.stereoconfig import stereoCamera 
from pathlib import Path
main_logger=logging.getLogger(__name__)
name = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]
#
@timethis
def combination(dataset,camera_config,ai_model,depth_model,
                ImgL,ImgR,im0s,
                ImgLabel,ratio,focal,baseline,pixel_size):
    distance=np.zeros((10,7),dtype=float)
    depth=np.zeros((1,),dtype=float) #distance calculate
    img_left,img_right,img_ai, img_raw=Image_Rectification(camera_config, ImgL, ImgR, im0sz=im0s, imgsz=ai_model.imgsz, stride=ai_model.stride)
    disparity=depth_model.run(img_left,img_right)
    pred=ai_model.detect(dataset,source=img_ai,source_rectified=img_raw,im0s=im0s,
                         iou_thres=0.45,conf_thres=0.25, 
                         classes=None,augment=False,agnostic_nms=False,
                         disparity=disparity,ratio=ratio,focal=focal,baseline=baseline,pixel_size=pixel_size,
                         debug=True)
    # for i,det in enumerate(pred):
    #     for j,obj in enumerate(det):
    #         cx,cy=((obj[0]+obj[2])/2).round(),((obj[1]+obj[3])/2).round()
    #         dx,dy=((obj[2]-obj[0])*ratio).round(),((obj[3]-obj[1])*ratio).round()
    #         depth=disparity_centre(cx, cy, dx, dy, disparity, focal, baseline, pixel_size)
    #         distance[j,:]=np.concatenate((obj.cpu().numpy(),depth),axis=0)
            # print('$OBJ_DEPTH',distance[j,0],distance[j,1],distance[j,2],distance[j,3],distance[j,4],name[int(distance[j,5])],distance[j,6],sep=',',end='*FC\n')
    # distance=np.asarray(distance[:j+1,:]).reshape(-1,7)
    # print('---------------Done---------------')
    return pred

def initial_platform():
    source, pipe, weights, device, imgsz=opt.source, opt.pipe, opt.weights, opt.device, opt.img_size
    if YOLOv5.count != 0:
        del ai_model
    ai_model = YOLOv5(weights, device, imgsz)
    if opt.SGBM:
        if SGBM.count != 0:
            del depth_model
        depth_model = SGBM()
    else:
        if BM.count != 0:
            del depth_model
        depth_model = BM()
    config = stereoCamera()
    if opt.webcam:
    #     pipe = find_cam('Lena3d')
    #     assert pipe != None, 'found No webcam'
        Stereo_Dataset = LoadStereoWebcam(pipe, ai_model.imgsz, ai_model.stride, UMat=opt.UMat)
    else:
        Stereo_Dataset = LoadStereoImages(source, ai_model.imgsz, ai_model.stride)
    return ai_model,depth_model,config, Stereo_Dataset

if __name__ == '__main__':
    t0=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images/indoor', help='integrated image from stereo camera')
    parser.add_argument('--pipe', type=str, default='0', help='real time camera')
    parser.add_argument('--webcam', action='store_true', help='Choose webcam as stereo images input')
    parser.add_argument('--ImgLabel', type=str, default='test', help='Images ID or Label')  # file/folder, 0 for webcam
    parser.add_argument('--ratio', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--save_path', type=str, default='runs/detect/exp/SGBM', help='camera focal length')
    parser.add_argument('--SGBM', action='store_true', help='Choose SGBM as stereo depth algorithm')
    parser.add_argument('--UMat', action='store_true', help='Choose UMat as image data format')
    # parser.add_argument('--pixel_size', type=float, default=0.05, help='camera pixel size')
    parser.add_argument('--weights', type=str, default='runs/train/exp9/weights/best.pt', help='YOLOv5 model weights')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)
    check_requirements()
    logging.basicConfig(level=logging.WARNING,force=True)
    source, ImgLabel, ratio=opt.source, opt.ImgLabel, opt.ratio
    ai_model,depth_model,camera_config, Stereo_Dataset = initial_platform()
    for path, image_left, image_right, im0s, vid_cap in Stereo_Dataset:
        distance = combination(Stereo_Dataset,
                               camera_config,
                               ai_model,
                               depth_model,
                               image_left,
                               image_right,
                               im0s,
                               ImgLabel,
                               ratio,
                               camera_config.focal_length,
                               camera_config.baseline,
                               camera_config.pixel_size)
        file_path=str(Path(opt.save_path).absolute())
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        if Stereo_Dataset.mode == 'video' or Stereo_Dataset.mode == 'webcam':
            i = Stereo_Dataset.frame
        elif Stereo_Dataset.mode == 'image':
            i = Stereo_Dataset.count
        with open(os.path.join(file_path,str(i)),'w') as f:
            for j in range(len(distance)):
                f.write(str(distance[j][0])+','+str(distance[j][1])+','+str(distance[j][2])+','+str(distance[j][3])+','+str(distance[j][4])+','+name[int(distance[j][5])]+','+str(distance[j][6])+'\n')
    logging.info(f'Done.({time.time()-t0:.3f}s)')
    
