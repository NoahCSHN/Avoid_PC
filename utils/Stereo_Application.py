import numpy as np
import argparse,logging,time
import cv2
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

class BM:
    'BM Algorithm class'
    count=0
    def __init__(self):
        t0 = time.time()
        BM.count += 1
        self.stereo = cv2.StereoBM_create(48, 9)
        self.stereo.setUniquenessRatio(40)
        self.stereo.setTextureThreshold(20)
        # self.stereo.setROI1(5)
        # self.stereo.setTextureThreshold(5)
        # self.stereo.setTextureThreshold(5)
        # self.stereo.setTextureThreshold(5)
        # self.stereo.setTextureThreshold(5)
        # self.stereo.setTextureThreshold(5)
        logging.info(f'\nBM Inital Done. ({time.time() - t0:.3f}s)')
        
    def __del__(self):
        class_name=self.__class__.__name__
        print (class_name,"release")
    
    def run(self,ImgL,ImgR):
        t0=time.time()
        ImgL = cv2.cvtColor(ImgL, cv2.COLOR_BGR2GRAY)
        ImgR = cv2.cvtColor(ImgR, cv2.COLOR_BGR2GRAY)        
        disparity = self.stereo.compute(ImgL,ImgR).astype(np.float32) / 16.0
        logging.info(f'\nBM Done. ({time.time() - t0:.3f}s)')
        return disparity
        
        
class SGBM:
    'SGBM Algorithm Class'
    count=0
    def __init__(self):
        t0 = time.time()
        SGBM.count += 1
        self.window_size = 3
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=3,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        logging.info(f'\nSGBM Inital Done. ({time.time() - t0:.3f}s)')
    
    def __del__(self):
        class_name=self.__class__.__name__
        print (class_name,"release")
    
    # @timethis
    def run(self,ImgL,ImgR):
        t0 = time.time()
        self.imgL = ImgL
        self.imgR = ImgR
        logging.info(f'Images Inital Done. ({time.time() - t0:.3f}s)')
        self.disparity = self.stereo.compute(self.imgL, self.imgR, False).astype(np.float32) / 16.0
        logging.info(f'SGBM Done. ({time.time() - t0:.3f}s)')
        return self.disparity

def disparity_centre(x_centre, y_centre, x_diff, y_diff, disparity,focal,baseline,pixel_size):
    p=[-2,-1,0,1,2]
    d=np.zeros((25,),dtype=float)
    dis_mean=np.zeros((1,),dtype=float) #disparity calculate
    depth=np.zeros((1,),dtype=float) #distance calculate
    for i in range(5):
        for j in range(5):
            nx,ny=(x_centre+p[i]*x_diff).cpu().numpy(),(y_centre+p[j]*y_diff).cpu().numpy()
            nx,ny=int(nx),int(ny)
            logging.debug(f'{disparity[ny,nx]==-1},{disparity[ny,nx]:.2f}')
            d.flat[5*i+j]=disparity[ny,nx]
    logging.debug(d)
    d=d[d!=-1]
    logging.debug(d)
    if len(d):
        dis_mean[0]=d.mean()
    else:
        dis_mean[0]=-1
    if dis_mean > 0:
        depth[0] = focal*baseline/dis_mean
    else:
        depth[0] = -1
    return depth

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


