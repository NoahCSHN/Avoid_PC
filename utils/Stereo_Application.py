import numpy as np
import argparse,logging,time,os
import cv2,math
from matplotlib import pyplot as plt
from utils.general import timethis

class Cursor:
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.ax.figure.canvas.draw()

class Stereo_Matching():
    """
    @description  : Stereo_Matching Algorithm class
    ---------
    @function  :
    -------
    """
    count=0
    def __init__(self,BM=False,filter=True,
                filter_lambda=8000.0,filter_sigma=1.5,
                filter_unira=5,
                numdisparity=64,mindis=0,block=0,TextureThreshold=5,
                prefiltercap=64,prefiltersize=9,prefiltertype=1,
                SpeckleWindowSize=50,speckleRange=2,disp12maxdiff=1,
                sf_path=""):
        self.BM = BM
        Stereo_Matching.count += 1
        self.filter_en = filter
        self.lamdba=filter_lambda
        self.sigma=filter_sigma
        self.unira=filter_unira
        if not self.BM:
            self.window_size = 3
            '''
            #The second parameter controlling the disparity smoothness. 
            # The larger the values are, the smoother the disparity is. 
            # P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. 
            # P2 is the penalty on the disparity change by more than 1 between neighbor pixels. 
            # The algorithm requires P2 > P1 . 
            # See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*blockSize*blockSize and 32*number_of_image_channels*blockSize*blockSize , respectively).
            '''            
            self.left_matcher = cv2.StereoSGBM_create(
                minDisparity=mindis,
                numDisparities=numdisparity-mindis,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize=block,
                P1=8 * 3 * self.window_size ** 2,   
                P2=32 * 3 * self.window_size ** 2,  
                disp12MaxDiff=1,
                uniquenessRatio=self.unira,
                speckleWindowSize=SpeckleWindowSize,
                speckleRange=speckleRange,
                preFilterCap=prefiltercap,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )
        else:
            self.left_matcher = cv2.StereoBM_create(numdisparity, block)
            self.left_matcher.setUniquenessRatio(self.unira)
            self.left_matcher.setTextureThreshold(TextureThreshold)
            self.left_matcher.setMinDisparity(mindis)
            self.left_matcher.setDisp12MaxDiff(disp12maxdiff)
            self.left_matcher.setSpeckleRange(speckleRange)
            self.left_matcher.setSpeckleWindowSize(SpeckleWindowSize)
            self.left_matcher.setBlockSize(block)
            self.left_matcher.setNumDisparities(numdisparity)
            self.left_matcher.setPreFilterCap(prefiltercap)
            self.left_matcher.setPreFilterSize(prefiltersize)
            self.left_matcher.setPreFilterType(prefiltertype)
            # self.left_matcher.setROI1(0)
            # self.left_matcher.setROI2(0)
            # self.left_matcher.setSmallerBlockSize(0)
        if self.filter_en:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.filter.setLambda(self.lamdba)
            self.filter.setSigmaColor(self.sigma)
        if sf_path != '':
            self.write_file(sf_path)    

    def write_file(self,path):
        self.sf = cv2.FileStorage()
        file_path = os.path.join(path,'stereo_config.xml')
        self.sf.open(file_path,cv2.FileStorage_WRITE)
        self.sf.write('datetime', time.asctime())
        if self.BM:
            self.sf.startWriteStruct('stereoBM',cv2.FileNode_MAP)
        else:
            self.sf.startWriteStruct('stereoSGBM',cv2.FileNode_MAP)
        self.sf.write('NumDisparities',self.left_matcher.getNumDisparities())
        self.sf.write('MinDisparity',self.left_matcher.getMinDisparity())
        self.sf.write('BlockSize',self.left_matcher.getBlockSize())
        self.sf.write('Disp12MaxDiff',self.left_matcher.getDisp12MaxDiff())
        self.sf.write('SpeckleRange',self.left_matcher.getSpeckleRange())
        self.sf.write('SpeckleWindowSize',self.left_matcher.getSpeckleWindowSize())
        self.sf.write('PreFilterCap',self.left_matcher.getPreFilterCap())
        self.sf.write('UniquenessRatio',self.left_matcher.getUniquenessRatio())
        if self.BM:
            self.sf.write('PreFilterSize',self.left_matcher.getPreFilterSize())
            self.sf.write('PreFilterType',self.left_matcher.getPreFilterType())
            self.sf.write('ROI1',self.left_matcher.getROI1())
            self.sf.write('ROI2',self.left_matcher.getROI2())
            self.sf.write('SmallerBlockSize',self.left_matcher.getSmallerBlockSize())
            self.sf.write('TextureThreshold',self.left_matcher.getTextureThreshold())
        else:
            self.sf.write('Mode',self.left_matcher.getMode())
        self.sf.endWriteStruct()
        if self.filter_en:
            self.sf.startWriteStruct('DisparityWLSFilter',cv2.FileNode_MAP)
            self.sf.write('ConfidenceMap',self.filter.getConfidenceMap())
            self.sf.write('DepthDiscontinuityRadius',self.filter.getDepthDiscontinuityRadius())
            self.sf.write('Lambda',self.filter.getLambda())
            self.sf.write('LRCthresh',self.filter.getLRCthresh())
            self.sf.write('ROI',self.filter.getROI())
            self.sf.write('SigmaColor',self.filter.getSigmaColor())
            self.sf.endWriteStruct()
        self.sf.release()

    def __del__(self):
        class_name=self.__class__.__name__
        print ('\n',class_name,"release")

    def run(self,ImgL,ImgR,Q,Queue,UMat=False,filter=True):
        """
        @description  :compute the disparity of ImgL and ImgR and put the disparity map to Queue
        ---------
        @param  : ImgL, Gray image taked by the left camera
        @param  : ImgR, Gray image taked by the right camera
        @param  : Queue, the data container of python API queue, used for data interaction between thread
        @param  : UMat, bool, if true, the data type is UMat(GPU), otherwise, the data type is UMat(CPU)
        @param  : filter, bool, if true, return the disparity map with post filter, otherwise, return the raw disparity map
        -------
        @Returns  : disparity, a Mat with the same shape as ImgL
        -------
        """
        t0=time.time()
        if not self.filter_en:
            if not self.BM:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).astype(np.float32) / 16.0
            else:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).astype(np.float32) / 16.0
                logging.info('\nSM Done. (%.2fs)',(time.time() - t0)) #cp3.5  
            color_3d = cv2.reprojectImageTo3D(disparity_left,Q).reshape(-1,ImgL.shape[1],3)
            # Queue.put((disparity_left,color_3d))
            return (disparity_left,color_3d)
        else:
            if not self.BM:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).get().astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL, False).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL, ImgR, False).astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL, False).astype(np.float32) / 16.0
            else:
                if UMat:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).get().astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL).get().astype(np.float32) / 16.0
                else:
                    disparity_left = self.left_matcher.compute(ImgL,ImgR).astype(np.float32) / 16.0
                    disparity_right = self.right_matcher.compute(ImgR, ImgL).astype(np.float32) / 16.0
                logging.info('\nSM Done. (%.2fs)',(time.time() - t0)) #cp3.5            
            disparity=self.filter.filter(disparity_left, ImgL, disparity_map_right=disparity_right)
            color_3d = cv2.reprojectImageTo3D(disparity,Q).reshape(-1,ImgL.shape[1],3)
            # Queue.put((disparity,color_3d))
            return (disparity,color_3d)

def disparity_centre(raw_box,ratio,disparity,depth_map,focal,baseline,pixel_size,mindisparity):
    """
    @description  : from disparity map get the depth prediction of the (x_centre,y_centre) point
    ---------
    @param  :
        raw_box: the coordinates of the opposite angle of the prediction box
        ratio: the distance between to centre point
        disparity: type array, disparity map
        depth_map: type array, depth map
        focal: focal length in pixel unit 
        baseline: baseline in mm unit
        pixel_size: pixel_size in mm unit
    -------
    @Returns  :
    -------
    """
    '''
    logic: if the pixel number in the box in smaller than 225,than calculate the whole box pixels and get the average, 
    otherwise, 
    '''        
    depth=[]
    #%%%% TODO: 分9个图像框
    # print(raw_box)
    dx,dy=int((raw_box[2]-raw_box[0])*ratio),int((raw_box[3]-raw_box[1])*ratio)
    if (dx == 0) and (dy == 0):
        # %% caculate every pixel in box and get the Median
        for i in range(raw_box[2]-raw_box[0]):
            # print('\ndisparity row:',end=' ')
            for j in range(raw_box[3]-raw_box[1]):
                # print(disparity[(raw_box[0]+i),(raw_box[1]+j)],end=',')
                # if disparity[(raw_box[0]+i),(raw_box[1]+j)] > -11:
                #     depth.append(disparity[(raw_box[0]+i),(raw_box[1]+j)])
                depth.append(depth_map[(raw_box[0]+i),(raw_box[1]+j)])
        # print(depth,end='\r')
    else:
        cx,cy=int((raw_box[0]+raw_box[2])/2),int((raw_box[1]+raw_box[3])/2)
        dw,dh=int((raw_box[2]-raw_box[0])/6),int((raw_box[3]-raw_box[1])/6)
        cxcy=[(cx-2*dw,cy-2*dh),(cx,cy-2*dh),(cx+2*dw,cy-2*dh),\
            (cx-2*dw,cy),(cx,cy),(cx+2*dw,cy),\
            (cx-2*dw,cy+2*dh),(cx,cy+2*dh),(cx+2*dw,cy+2*dh)]
        # print(cxcy)
        # print(dx,dy)    

        #%%%% TODO: 每个框计算深度均值  
        for x_centre,y_centre in cxcy:
            p=[-2,-1,0,1,2]
            d=np.zeros((25,),dtype=float)
            dis_mean=0.
            for i in range(5):
                for j in range(5):
                    nx,ny=int(x_centre+p[i]*dx),int(y_centre+p[j]*dy)
                    # print('(%d,%d)'%(nx,ny),end=' ')
                    # d.flat[5*i+j]=disparity[ny,nx]
                    d.flat[5*i+j]=depth_map[ny,nx]
            d=d.ravel()
            if mindisparity < 0:
                d=d[d>(mindisparity-1.)]
            else:
                d=d[d>-1.]
            d=np.sort(d,axis=None)
            # print(d,end='\r')
            if len(d) >= 5:
                d=np.delete(d,[0,-1])
                dis_mean = d.mean()
                depth.append(dis_mean)
    # %%%% TODO: 取众多框计算值的中位数 
    depth = np.abs(depth)
    depth.sort()
    if len(depth) == 0:
        temp_dis = -1
    elif (len(depth)%2 == 0) & (len(depth)>1):
        if (depth[math.floor(len(depth)/2)] != 0) and (depth[math.floor(len(depth)/2)-1] != 0):
            # temp_dis = ((focal*baseline/abs(depth[math.floor(len(depth)/2)]))+(focal*baseline/abs(depth[math.floor(len(depth)/2)-1])))/2
            temp_dis = (depth[math.floor(len(depth)/2)] + depth[math.floor(len(depth)/2)-1])/2
        else:
            temp_dis = -1
    else:
        if depth[math.floor(len(depth)/2)] != 0:
            # temp_dis = focal*baseline/abs(depth[math.floor(len(depth)/2)])
            temp_dis = depth[math.floor(len(depth)/2)]
        else:
            temp_dis = -1
    return temp_dis

def stereo_sgbm(ImgLPath='../data/images/left.png',ImgRPath='../data/images/right.png', path=True):
    t0 = time.time()
    imgL = cv2.imread(ImgLPath)
    imgR = cv2.imread(ImgRPath)
    logging.info(f'Images Inital Done. ({time.time() - t0:.3f}s)')
    # disparity range tuning
    window_size = 3
    # min_disp = 0
    # num_disp = 320 - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=3,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    logging.info(f'SGBM Done. ({time.time() - t0:.3f}s)')
    return disparity

def detect_disparity(ImgLPath='../data/images/Left1_rectified.bmp',ImgRPath='../data/images/Right1_rectified.bmp'):
    imgL = cv2.imread(ImgLPath)
    imgR = cv2.imread(ImgRPath)
    # disparity range tuning
    window_size = 3
    # min_disp = 0
    # num_disp = 320 - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=3,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    fig, ax = plt.subplots()
    plt.imshow(disparity, 'gray')
    cursor = Cursor(ax)
    fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_source', type=str, default='data/images/left.png', help='source')
    parser.add_argument('--right_source', type=str, default='data/images/left.png', help='source')
    opt = parser.parse_args()
    print(opt)    
    detect_disparity()


