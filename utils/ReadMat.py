'''
Author: Noah
Date: 2021-09-15 16:18:42
LastEditTime: 2021-09-15 17:48:23
LastEditors: Please set LastEditors
Description: Read stereo camera calibration mat and write it in stereo_calibration configuration file.
FilePath: /AI_SGBM/utils/ReadMat.py
'''
import scipy.io as scio
import os

def switch():

if __name__ == '__main__':
    # %%
    #TODO:read Mat file
    Matfile_path = os.walk("/home/bynav/0_code/AI_SGBM/data/calibration/20210915/20210915161059")
    for path,dir_list,file_list in Matfile_path:
        print(f'{path},{dir_list},{file_list}')
        assert file_list,f"no calibration file in the directory: {path}"
        calib_data={}
        for mat_file in file_list:
            key = mat_file.split('.')
            print(mat_file)
            calib_data[key] = scio.loadmat(os.path.join(path,mat_file))
        print(calib_data)
        break

    # %%
    #TODO:write to the stereo_calibration configuration file
