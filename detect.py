import argparse,logging,time,os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.Stereo_Application import disparity_centre 
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, timethis
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#%% YOLOv5 interface for other project
class YOLOv5:
    'YOLOv5 Algorithm Class'
    count=0
    #%%% initialization for pretrained YOLOv5 model
    def __init__(self,weights,device,imgsz):
        t0=time.time()
        YOLOv5.count += 1
        self.weights = weights
        self.device = select_device(device)
        self.half = self.device.type!='cpu'
        self.model = attempt_load(self.weights,map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz,s=self.stride)
        if self.half:
            self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        logging.info(f'YOLOv5 Inital Done. ({time.time() - t0:.3f}s)')
    
    #%%% release of model to avoid repeat instantiation 
    def __del__(self):
        class_name=self.__class__.__name__
        print (class_name,"release")
    
    #%%% object detect  
    # @timethis  
    def detect(self,
               source,source_rectified,im0s,iou_thres,conf_thres,
               classes,augment,agnostic_nms,
               queue,debug=False):
        # disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=256/128), cv2.COLORMAP_JET)
        # distance=np.zeros((10,7),dtype=float)
        # depth=np.zeros((1,),dtype=float) #distance calculate
        img = source
        if debug:
            # im0 = img.transpose(1,2,0)
            # im0 = im0[:,:,::-1]
            im0 = source_rectified
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            
        # Inference
        pred = self.model(img, augment=augment)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        # pred = pred.numpy()
        # t2 = time_synchronized()
        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)        
        # queue.put((pred,names))
        return (pred,names)
   

#%% main YOLOv5 implementation
def YOLO_detect(save_img=False,
           source='data/images/left.png',
           weights='runs/train/exp9/weights/best.pt',
           view_img=False,
           save_txt=False,
           imgsz=640,
           process_device='0',
           augment=False,
           conf_thres=0.25,
           iou_thres=0.45,
           classes=None,
           agnostic_nms=False
           ):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(process_device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
    # Process detections
        # for i, det in enumerate(pred):  # detections per image
        #     if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
    #             result.append(det)
    # logging.info(f'Detect Done. ({time.time() - t0:.3f}s)')
    # return result # (x1,y1,x2,y2,conf,cls)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

#%% independent main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp9/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images/Left1_rectified.bmp', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                YOLO_detect(source=opt.source, 
                       weights=opt.weights, 
                       view_img=opt.view_img, 
                       save_txt=opt.save_txt, 
                       imgsz =opt.img_size,
                       process_device=opt.device,
                       augment=opt.augment,
                       conf_thres=opt.conf-thres,
                       iou_thres=opt.iou-thres,
                       classes=opt.classes,
                       agnostic_nms=opt.agnostic_nms
                       )
                strip_optimizer(opt.weights)
        else:
            YOLO_detect(source=opt.source, 
                   weights=opt.weights, 
                   view_img=opt.view_img, 
                   save_txt=opt.save_txt, 
                   imgsz =opt.img_size,
                   process_device=opt.device,
                   augment=opt.augment,
                   conf_thres=opt.conf_thres,
                   iou_thres=opt.iou_thres,
                   classes=opt.classes,
                   agnostic_nms=opt.agnostic_nms)
