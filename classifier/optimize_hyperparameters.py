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

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.algorithms.hyperparameters import sta

from pymoo.optimize import minimize

def cost_function(img):
    global clahe
    # print(clahe.getTilesGridSize())
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

    return Parallel(n_jobs=-1, prefer="threads")(delayed(cost_function)(inp) for inp in test_images)

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         xl=np.array([0.0000001, 4]),
                         xu=np.array([10,50]))

    def _evaluate(self, x, out, *args, **kwargs):
        global p
        global clahe
        global loaded_images
        clip_limit = x[0]
        tile_size = int(x[1])

        print(f'({clip_limit}, {tile_size})')
        
        res = parallel_cost_function(loaded_images, clip_limit, tile_size)

        list_entropy = [item[0] for item in res]
        list_ssim = [item[1] for item in res]
        f1 = -np.mean(list_entropy)
        f2 = -np.mean(list_ssim)

        out["F"] = [f1, f2]
    
if __name__ == '__main__':

    img_dir = "/mnt/external.data/TowbinLab/kstojanovski/20211206_Ti2_10x_wBT264_186_daf-16d_25C_20211206_170249_693/analysis/ch2/"

    images = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
    nb = 64
    test_images = random.sample(images, nb)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(6,6))
    loaded_images = []
    for img in test_images:
        loaded_images.append(ti.imread(img))

    problem = MyProblem()

    ref_dirs = np.array([[1.0]])
    algorithm = UNSGA3(
        ref_dirs,
        pop_size=100,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 50)

    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)

    X = res.X
    F = res.F

    np.savez("./optimization_result.npz", X=X, F=F)