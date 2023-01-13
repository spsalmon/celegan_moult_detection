import numpy as np
import os
import tifffile as ti
import cv2
import random
import torch 
import skimage
import matplotlib
import timeit

from pymoo.core.problem import ElementwiseProblem
from skimage.metrics import structural_similarity as ssim
import skimage.measure    


from skimage.metrics import structural_similarity as ssim
import skimage.measure    

import sys
import time
import traceback
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from skimage.io import imread

from itertools import repeat

def cost_function(img):
    global clahe
    improved_img = clahe.apply(img)
    entropy = skimage.measure.shannon_entropy(improved_img)
    struct_similarity = ssim(img, improved_img)
    return [entropy, struct_similarity]

def serial_cost_function(test_images, clip_limit, tile_size):
    global clahe
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size,tile_size))

    res = []
    for inp in test_images:
        res.append(cost_function(inp))
    return res

def parallel_cost_function(test_images, clip_limit, tile_size):
    global clahe
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size,tile_size))

    # return Parallel(n_jobs=-1, backend="multiprocessing")(delayed(cost_function)(inp) for inp in test_images)
    return Parallel(n_jobs=-1, prefer="threads")(delayed(cost_function)(inp) for inp in test_images)
    
if __name__ == '__main__':

    img_dir = "/mnt/external.data/TowbinLab/kstojanovski/20211206_Ti2_10x_wBT264_186_daf-16d_25C_20211206_170249_693/analysis/ch2/"

    images = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
    nb = 100
    test_images = random.sample(images, nb)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(6,6))
    loaded_images = []
    for img in test_images:
        loaded_images.append(ti.imread(img))
    clip_limit = 0.6
    tile_size = 4

    # parallel_cost_function(test_images, clip_limit, tile_size)
    time = timeit.repeat(lambda:parallel_cost_function(loaded_images, clip_limit, tile_size), repeat = 10, number=1)
    print(time)
    print('first execution:', time[0]/nb, 's')
    print('mean execution time :', np.mean(time)/nb, 's')
    print('full execution time :', np.mean(time))