'''
Author: your name
Date: 2021-04-08 16:03:30
LastEditTime: 2021-09-17 15:39:14
LastEditors: Please set LastEditors
Description: the user interface of avoid model
FilePath: /AI_SGBM/combination.py
'''

import argparse,logging,time,os,cv2,queue,sys
import numpy as np
from utils.Stereo_Application import Stereo_Matching,disparity_centre
from detect import YOLOv5
from utils.datasets import LoadStereoImages, LoadStereoWebcam, DATASET_NAMES
from utils.general import confirm_dir,check_requirements,scale_coords,timethis,camera_mode,socket_client,video_writer
from utils.img_preprocess import Image_Rectification
from utils.plots import plot_one_box
from models.stereoconfig import stereoCamera 
from threading import Thread
from pathlib import Path
from datetime import datetime

main_logger=logging.getLogger(__name__)
name = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]
#%%
# @timethis
def image_process(camera_config,ai_model,depth_model,
                  ImgL,ImgR,im0s,
                  ImgLabel,ratio,
                  disparity_queue,pred_queue):
    
    j=0
    i=0
    distance=np.zeros((20,11),dtype=float)
    # depth=np.zeros((1,),dtype=float) #distance calculate
    img_left, img_right, img_ai, img_raw=Image_Rectification(camera_config, ImgL, ImgR, im0sz=im0s, imgsz=ai_model.imgsz, stride=ai_model.stride,UMat=opt.UMat, debug=opt.debug)
    disparity, color_3d = depth_model.run(img_left,img_right,camera_config.Q,disparity_queue,opt.UMat,opt.filter)
    pred, names = ai_model.detect(img_ai,img_raw,im0s,0.45,0.25, None,False,False, pred_queue,opt.debug)
    # sm_t = Thread(target=depth_model.run,args=(img_left,img_right,camera_config.Q,disparity_queue,opt.UMat,opt.filter))
    # ai_t = Thread(target=ai_model.detect,args=(img_ai,img_raw,im0s,0.45,0.25, 
    #                                            None,False,False,
    #                                            pred_queue,opt.debug))
    # sm_t.start()
    # ai_t.start()
    # sm_t.join()
    # ai_t.join()
    # Process detections
    # disparity, color_3d = disparity_queue.get()
    # pred, names = pred_queue.get()
    for i, det in enumerate(pred):  # detections per image
#%% TODO: 将一张图片的预测框逐条分开，并且还原到原始图像尺寸
        if len(det):
            # Rescale boxes from img_size to im0 size
            det_resize = det.clone().detach()
            # print(det_resize.shape)
            # print(det_resize)
            # det_resize = np.copy(det)
            det_resize[:,:4] = scale_coords(img_ai.shape, det_resize[:, :4], img_raw.shape).round()
            for j,obj in enumerate(det):
                temp_dis=disparity_centre(obj, ratio, disparity, color_3d[:,:,2],camera_config.focal_length, camera_config.baseline, camera_config.pixel_size, opt.sm_mindi)
#%% TODO: 将最终深度结果画到图像里
                # if debug:
                xyxy = [det_resize[j,0],det_resize[j,1],det_resize[j,2],det_resize[j,3]]
                label = f'{names[int(obj[5])]} {obj[4]:.2f}:{temp_dis:.2f}'
                plot_one_box(xyxy, img_raw, label=label, color=DATASET_NAMES.name_color[DATASET_NAMES.coco_names.index(names[int(obj[5])])], line_thickness=2)
                    # xyxy = [int((det_resize[j,0]+det_resize[j,2])/2)-2*int((det_resize[j,2]-det_resize[j,0])*ratio),\
                    #         int((det_resize[j,1]+det_resize[j,3])/2)-2*int((det_resize[j,3]-det_resize[j,1])*ratio),\
                    #         int((det_resize[j,0]+det_resize[j,2])/2)+2*int((det_resize[j,2]-det_resize[j,0])*ratio),\
                    #         int((det_resize[j,1]+det_resize[j,3])/2)+2*int((det_resize[j,3]-det_resize[j,1])*ratio)]
                    # label = f'{depth[0]:.2f}'
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(obj[5])], line_thickness=1)
                distance[j,0] = temp_dis
                distance[j,1:7] = det_resize[j,:].cpu()
                distance[j,7:] = obj[:4].cpu()
    if j == 0:
        distance = []
    else:
        distance = distance[:j,:]
    return distance,img_raw # (distance,x1,y1,x2,y2,conf,cls)

#%% 
def result_handle(v_writer = '',dataset = '',distance = '',image = '',soc_client = '',save_path = ''):
    cv2.namedWindow('result',flags=cv2.WINDOW_NORMAL)
    if opt.debug and dataset.mode == 'image':
        file_path = confirm_dir(save_path,"proc_image")
        files = os.path.join(file_path,str(dataset.count)+'.bmp')
        cv2.imwrite(files, image)
    elif opt.debug and (dataset.mode == 'video' or dataset.mode == 'webcam'):
        v_writer.write(image)
    cv2.imshow('result',image)
    if cv2.waitKey(1) == ord('q'):
        sys.exit("manual exit")
    if dataset.mode == 'video' or dataset.mode == 'webcam':
        i = dataset.frame
    elif Stereo_Dataset.mode == 'image':
        i = dataset.count
    with open(os.path.join(save_path,"result.txt"),'a+') as f:
        f.write("Frame "+str(i)+":")
        if len(distance):
            for j in range(len(distance)):
                f.write("\n  "+name[int(distance[j][5])]
                +': distance,'+str(distance[j][0])
                +'; x0,'+str(distance[j][1])
                +'; y0,'+str(distance[j][2])
                +'; x1,'+str(distance[j][3])
                +'; y1,'+str(distance[j][4])
                +','+str(distance[j][7])
                +','+str(distance[j][8])
                +','+str(distance[j][9])
                +','+str(distance[j][10])
                +'\n')
        else:
            f.write("None detected.\n")
    # cv2.destroyAllWindows()
    logging.info(f'Done.({time.time()-t0:.3f}s)')
    

#%%
def initial_platform(save_path):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    
    # initialize information container for information exchange between mulit-threads
    disparity_queue = queue.Queue(maxsize=1)
    pred_queue = queue.Queue(maxsize=1)
    
    # image transfer tcp connect with another user
    if opt.remote:
        soc_client=socket_client(address=(opt.tcp_ip,opt.tcp_port))
    else:
        soc_client=None

    # initialize AI and stereo matching models
    if YOLOv5.count != 0:
        del ai_model
    ai_model = YOLOv5(opt.weights, opt.device, opt.img_size)
    if Stereo_Matching.count != 0:
        del depth_model
    depth_model = Stereo_Matching(opt.BM, opt.filter,\
                                  opt.sm_lambda, opt.sm_sigma, opt.sm_UniRa,\
                                  opt.sm_numdi, opt.sm_mindi, opt.sm_block, opt.sm_tt,\
                                  opt.sm_pfc, opt.sm_pfs, opt.sm_pft,\
                                  opt.sm_sws, opt.sm_sr, opt.sm_d12md, save_path)
    
    # camera mode contains: image size, camera recitfication parameters 
    cam_mode = camera_mode(opt.cam_type)
    config = stereoCamera(mode=cam_mode.mode.value,height=cam_mode.size[1],width=cam_mode.size[0])

    # load image source to iterator
    if opt.webcam:
        Stereo_Dataset = LoadStereoWebcam(opt.pipe, opt.fps, ai_model.imgsz, ai_model.stride, UMat=opt.UMat)
    else:
        Stereo_Dataset = LoadStereoImages(opt.source, ai_model.imgsz, ai_model.stride,opt.save_path)
    return ai_model, depth_model, config, Stereo_Dataset, disparity_queue, pred_queue, soc_client

#%%
if __name__ == '__main__':
    t0=time.time()
    parser = argparse.ArgumentParser()
    # camera parameters configuration
    parser.add_argument('--source', type=str, default='data/images/indoor', help='integrated image from stereo camera')
    parser.add_argument('--pipe', type=str, default='0', help='real time camera')
    parser.add_argument('--webcam', action='store_true', help='Choose webcam as stereo images input')
    parser.add_argument("--fps", help="The webcam frequency", type=int, default=4)
    # YOLOv5 parameters configuration
    parser.add_argument('--ImgLabel', type=str, default='test', help='Images ID or Label')  # file/folder, 0 for webcam
    parser.add_argument('--ratio', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='YOLOv5 model weights')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    # SGBM or BM parameters configuration
    parser.add_argument('--BM', action='store_true', help='Choose BM as stereo depth algorithm')
    parser.add_argument('--UMat', action='store_true', help='Choose UMat as image data format')
    parser.add_argument("--tcp_port", help="tcp port", type=int, default=9191)
    parser.add_argument("--tcp_ip", help="tcp ip", type=str, default='192.168.2.75')
    parser.add_argument("--out_range", help="The data size for model input", nargs='+', type=float, default=[0.3,1])
    parser.add_argument("--sm_lambda", help="Stereo matching post filter parameter lambda", type=float, default=8000)
    parser.add_argument("--sm_sigma", help="Stereo matching post filter parameter sigmacolor", type=float, default=2.0)
    parser.add_argument("--sm_UniRa", help="Stereo matching post filter parameter UniquenessRatio", type=int, default=5)
    parser.add_argument("--sm_numdi", help="Stereo matching max number disparity", type=int, default=64)
    parser.add_argument("--sm_mindi", help="Stereo matching min number disparity", type=int, default=-5)
    parser.add_argument("--sm_block", help="Stereo matching blocksize", type=int, default=9)
    parser.add_argument("--sm_tt", help="Stereo matching blocksize", type=int, default=5)        
    parser.add_argument("--sm_pfc", help="Stereo matching PreFilterCap", type=int, default=63)    
    parser.add_argument("--sm_pfs", help="Stereo matching PreFilterSize", type=int, default=9)    
    parser.add_argument("--sm_pft", help="Stereo matching PreFilterType", type=int, default=1)    
    parser.add_argument("--sm_sws", help="Stereo matching SpeckleWindowSize", type=int, default=50)  
    parser.add_argument("--sm_sr", help="Stereo matching SpeckleRange", type=int, default=2)    
    parser.add_argument("--sm_d12md", help="Stereo matching Disp12MaxDiff", type=int, default=1)    
    parser.add_argument("--filter", help="Enable post WLS filter",action="store_true")
    parser.add_argument("--cam_type", help="0: OV9714, 1: AR0135 1280X720; 2: AR0135 1280X960; 3:AR0135 416X416; 4:AR0135 640X640; 5:AR0135 640X480; 6:MIDDLEBURY 416X360", type=int, default=4)
    parser.add_argument("--stereo_ratio", help="ratio for distance calculate", type=float, default=0.05)
    # file and log configuration
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--debug", help="save data source for replay", action="store_true")
    parser.add_argument("--visual", help="result visualization", action="store_true")
    parser.add_argument("--remote", help="process image on a remote controller", action="store_true")
    parser.add_argument("--save_result", help="inference result save", action="store_true")
    parser.add_argument("--save_path",help="path for result saving",type=str,default="runs/detect/")    
    opt = parser.parse_args()
    print(opt)
    check_requirements()
    save_path = confirm_dir(opt.save_path,opt.ImgLabel)
    save_path = confirm_dir(save_path,datetime.now().strftime("%Y%m%d"))
    save_path = confirm_dir(save_path,datetime.now().strftime("%H%M%S"))
    if opt.verbose:
        logging.basicConfig(filename=os.path.join(save_path,'log.txt'),
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING,force=True)
#%%
    ai_model,depth_model,camera_config, Stereo_Dataset, disparity_queue, pred_queue, socket_client = initial_platform(save_path)
    if opt.debug:
        w,h,fps = Stereo_Dataset.get_camera_config()
        v_writer = video_writer(w,h,fps,save_path)
    for image_left, image_right, im0s, _ in Stereo_Dataset:
        distance, image = image_process(camera_config,
                                        ai_model,
                                        depth_model,
                                        image_left,
                                        image_right,
                                        im0s,
                                        opt.ImgLabel,
                                        opt.ratio,
                                        disparity_queue,
                                        pred_queue)
        if opt.debug:
            result_handle(v_writer,Stereo_Dataset,distance,image,socket_client,save_path)
        else:
            result_handle(dataset = Stereo_Dataset,distance = distance,image = image,soc_client = socket_client,save_path = save_path)