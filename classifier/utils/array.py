import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import skimage

def convert_to_one_hot_encode(tensor, n_classes):
        np_array = tensor.cpu().numpy()
        one_hot = np.empty([np_array.shape[0], n_classes, np_array.shape[1], np_array.shape[2]])

        for i in range(n_classes):
            one_hot[:, i, :, :] = np_array == i
        
        return torch.tensor(one_hot.astype(np.float))