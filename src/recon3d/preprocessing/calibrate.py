import os
from pathlib import Path
import argparse

import cv2 # NOTE: recommand to use opencv 4.3.0.38 !!!!
import numpy as np
# from matplotlib import pyplot as plt

import multiprocessing
import multiprocessing.dummy
from functools import partial


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkerboard_dir', type=str, help='the directory to the checkerboard image')
    parser.add_argument('--checkerboard', nargs='+', type=int) # e.g. --checkerboard 6 9
    parser.add_argument('--k_d_npz', type=str)
    parser.add_argument('--result_dir', type=str)

    args = parser.parse_args()

    return args


def get_K_and_D(checkerboard, imgFileList):
    print('Start calculating camera parameter...')
    CB = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    
    objp = np.zeros((1, CB[0]*CB[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CB[0], 0:CB[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] 
    imgpoints = [] 
    

    for idx, fname in enumerate(imgFileList):
        img = cv2.imread(fname)
        
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # plt.figure()
        # plt.imshow(gray, cmap='gray')
        # plt.show()
        ret, corners = cv2.findChessboardCorners(gray, CB,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # ret, corners = cv2.findChessboardCornersSB(gray, CB, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            print('[ {:2d} ] Find chess corner of image: {}, ret: {}, corners: {}'.format(idx, fname, ret, corners.shape))
        else:
            print('[ {:2d} ] Find chess corner of image: {}, ret: {}'.format(idx, fname, ret))
        # show = cv2.drawChessboardCorners(gray, CB, corners, ret)
        # cv2.imwrite('/volume/cpl-dev/sfm/hsien/insta360pro2_calib/debug/'+str(ret)+'_'+fname.split('/')[-1], show)

    # NOTE: recommand to use opencv 4.3.0 !!!!
    while True:
        assert len(objpoints) > 0, "There are no valid images from which to calibrate."
        try:
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rms, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=objpoints,
                imagePoints=imgpoints,
                image_size=gray.shape[::-1],
                K=K,
                D=D,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            )
            print('Found a calibration based on {} well-conditioned images.'.format(len(objpoints)))
            break
        except cv2.error as err: # NOTE: recommand to use opencv 4.3.0 !!!!
            try:
                idx = int(err.msg.split('array ')[1][0])  # Parse index of invalid image from error message
                objpoints.pop(idx)
                imgpoints.pop(idx)
                print("Removed ill-conditioned image {} from the data.  Trying again...".format(idx))
            except IndexError:
                raise err

    DIM = _img_shape[::-1]
    
    print('DIM: {}\nK = {}\nD = {}'.format(DIM, K, D))

    return DIM, K, D




def defisheye(DIM, K, D, filename_list, input_dir, output_dir):
    def parallel_task(filename):
        img = cv2.imread(os.path.join(input_dir, filename))
        if img is None:
            return
        img = cv2.fisheye.undistortImage(img, K, D, Knew=K, new_size=DIM)
        print(os.path.join(output_dir, filename))
        cv2.imwrite(os.path.join(output_dir, filename), img)
    pool = multiprocessing.dummy.Pool(16)
    pool.map(parallel_task, filename_list) 

    return 0 

def main():
    # 0. check 
    
    checkerboard_dir = Path(args.checkerboard_dir)
    result_dir = Path(args.result_dir)
    
    print('Argument:\ncheckerboard_dir: {}\ncheckerboard: {}\nk_d_npz: {}\nresult_dir: {}\n'.format(checkerboard_dir, args.checkerboard, args.k_d_npz, args.result_dir))

    # input checkerboard img list
    # imgFileList = get_img_file_list(checkerboard_dir, args.coverage)
    imgFileList = list(map(str, checkerboard_dir.glob('*.jpg')))
    # get camera intrinsic parameter K and D
    DIM, K, D = get_K_and_D(tuple(args.checkerboard), imgFileList)
    np.savez(args.k_d_npz, DIM=DIM, K=K, D=D)
    defisheye(DIM, K, D, os.listdir(checkerboard_dir), checkerboard_dir, result_dir)


if __name__ == "__main__":
    args = get_args()
    main()