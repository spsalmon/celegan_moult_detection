import numpy as np
import os
import tifffile as ti
import cv2
import random
import torch 
import skimage
import matplotlib
import pathos.pools as pp
from pymoo.core.problem import ElementwiseProblem
from skimage.metrics import structural_similarity as ssim
import skimage.measure    


from skimage.metrics import structural_similarity as ssim
import skimage.measure    


matplotlib.rcParams['figure.figsize'] = [13, 13]

img_dir = "/mnt/external.data/TowbinLab/kstojanovski/20211206_Ti2_10x_wBT264_186_daf-16d_25C_20211206_170249_693/analysis/ch2/"

images = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]

# clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4, 4))
# p = pp.ProcessPool(32)

# # test_images = random.sample(images, 20)
test_images = random.sample(images, 100)

data = []

for image in test_images:
    img = ti.imread(image)
    data.append(img)

while True:
    print("0")

# def cost_function(image):
#     global clahe
#     print(clahe.getTilesGridSize())
#     img = ti.imread(image)
#     improved_img = clahe.apply(img)
#     entropy = skimage.measure.shannon_entropy(improved_img)
#     struct_similarity = ssim(img, improved_img)
#     # return None

#     # return [entropy, struct_similarity]
#     return [entropy, struct_similarity]


# class MyProblem(ElementwiseProblem):

#     def __init__(self):
#         super().__init__(n_var=2,
#                          n_obj=2,
#                          xl=np.array([0.0000001, 8]),
#                          xu=np.array([1,128]))

#     def _evaluate(self, x, out, *args, **kwargs):
#         global p
#         global clahe
#         clip_limit = x[0]
#         tile_size = int(x[1])

#         clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
#         res = p.map(cost_function, test_images)

#         list_entropy = [item[0] for item in res]
#         list_ssim = [item[1] for item in res]
#         f1 = -np.mean(list_entropy)
#         f2 = -np.mean(list_ssim)

#         out["F"] = [f1, f2]

# problem = MyProblem()

# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
# from pymoo.operators.sampling.rnd import FloatRandomSampling
# from pymoo.termination import get_termination

# algorithm = NSGA2(
#     pop_size=40,
#     n_offsprings=10,
#     sampling=FloatRandomSampling(),
#     crossover=SBX(prob=0.9, eta=15),
#     mutation=PM(eta=20),
#     eliminate_duplicates=True
# )

# termination = get_termination("n_gen", 100)


# from pymoo.optimize import minimize

# res = minimize(problem,
#                algorithm,
#                termination,
#                seed=1,
#                save_history=True,
#                verbose=True)

# X = res.X
# F = res.F

# np.savez("./optimization_result.npz", X=X, F=F)