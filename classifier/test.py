import argparse
import logging
from lzma import PRESET_DEFAULT
import os
import sys
from xml.etree.ElementPath import prepare_descendant

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from yaml import load

from utils.loss_function import *
from utils.accuracy import *
from utils.dataset import *
from utils.array import *

import argparse
import logging
from lzma import PRESET_DEFAULT
import os
import sys
from xml.etree.ElementPath import prepare_descendant

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model import ResNet50, ResNet101, ResNet152
from utils.dataset import *
from utils.array import *
from scipy import interpolate

from torch.utils.tensorboard.writer import SummaryWriter
from utils.dataset import TrainingDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image, ImageFilter
from collections import OrderedDict

import pandas as pd

import time

def load_model(model_path: str, device: torch.device, parallel : bool) -> Tuple[nn.Module, int, int, int, list, float, str]:
    """
    Loads the model from the given path and returns it as a DataParallel model, along with other relevant information.

    Parameters:
        model_path (str): The path to the model checkpoint.
        device (torch.device): The device to load the model onto.
        parallel (bool): If true, will wrap the loaded model in a DataParallel module.

    Returns:
        tuple: A tuple containing the following elements:
            - nn.Module: The loaded model wrapped in a DataParallel module.
            - int: The epoch at which the model was trained.
            - int: The size of the model input.
            - int: The number of channels in the model input.
            - int: The number of classes in the model output.
            - float: The IoU (intersection over union) value for the model.
    """
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Load the checkpoint from the given path and extract relevant information
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    epoch = checkpoint['epoch']
    loss_val = checkpoint['loss_val']
    accuracy_val = checkpoint['accuracy_val']
    layers = checkpoint['layers']
    dim = checkpoint['dim']
    classes = checkpoint['classes']
    n_channels = checkpoint['n_channels']

    # Initializing the neural network
    if layers == 50:
        network_name = "ResNet50"
        net = ResNet50(len(classes), n_channels)
    elif layers == 101:
        network_name = "ResNet101"
        net = ResNet101(len(classes), n_channels)
    elif layers == 152:
        network_name = "ResNet152"
        net = ResNet152(len(classes), n_channels)
    else:
        raise ValueError("The number of layers can only be 50, 101 or 152.")

    # Create a new ordered dictionary to store the state dictionary without the `module.` prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    logging.info(f'Model loaded from {model_path}')

    del state_dict
    torch.cuda.empty_cache()

    # Wrap the model in a DataParallel module if parallel is True
    if parallel:
        net = nn.parallel.DataParallel(net)

    # Move model to the chosen device
    net.to(device=device)

    # Return network and relevant information about it
    return net, epoch, dim, n_channels, classes, accuracy_val, network_name


def get_dim_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    return int(checkpoint['dim'])

def test_model(net, model_name,
              device,
              batch_size=1,
              dim=512):


    dataset_test = TrainingDataset(img_dir, labels_csv, classes, dim=dim)
    n_test = len(dataset_test)
    test_loader = DataLoader(dataset_test, batch_size=batch_size,
                             shuffle=False, num_workers=32, pin_memory=True)

    TN = 0
    TP = 0
    FN = 0
    FP = 0

    with torch.no_grad():
        with tqdm(total=n_test, desc=f'Testing of {model_name}', unit='img') as pbar:
            for batch in test_loader:
                imgs = batch['image']
                true_labels = batch['label']

                imgs = imgs.to(device=device, dtype=torch.float32)

                probs = torch.softmax(net(imgs), dim=1)

                del imgs
                if (len(probs.shape) == 1):
                    probs = probs.unsqueeze(1)

                for i in range(probs.shape[0]):
                    pred = probs[i, :]
                    true_pred = true_labels[i]
                    max_idx = torch.argmax(pred, keepdim=True)
                    label = classes[max_idx]

                    true_max_idx = torch.argmax(true_pred, keepdim=True)
                    true_label = classes[true_max_idx]

                    if label == "not_moulting" and true_label == "not_moulting":
                        # print("TN, :", TN)
                        TN += 1
                    elif label == "not_moulting" and true_label == "moulting":
                        # print("FN :", FN)
                        FN += 1
                    elif label == "moulting" and true_label == "not_moulting":
                        # print("FP :", FP)
                        FP += 1
                    elif label == "moulting" and true_label == "moulting":
                        # print("TP :", TP)
                        TP += 1

                del probs
                torch.cuda.empty_cache()
                pbar.update(batch_size)

        return TN, TP, FN, FP

def performance_measures(TN, TP, FN, FP):

    recall = TP/(TP + FN)
    fpr = 1 - (TN/(TN + FP))
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    error_rate = (1 - accuracy)
    precision = TP/(TP + FP)
    f_measure = 2/((1/precision) + (1/recall))
    f_beta = (1 + 0.5**2)*((precision*recall)/((0.5**2)*precision + recall))

    return recall, fpr, accuracy, error_rate, precision, f_measure, f_beta


def get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Predict moults an experiment folder',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add the arguments to the parser
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=6,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-i', '--images-dir', type=str, required=True,
                        help='Directory containing the test images', dest='test_dir')
    parser.add_argument('-o', '--report-dir', type=str, required=True,
                        help='Report directory', dest='report_dir')
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help='Path to the model', dest='model_path')

    # Parse the arguments and return the resulting namespace object
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    batch_size = args.batch_size
    model = args.model_path
    test_report_dir = args.report_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet, epoch, dim, n_channels, classes, accuracy_val, network_name = load_model("./checkpoints/very_good_for_moults.pth", device, torch.cuda.device_count() > 1)

    output_dataframe = pd.DataFrame(columns=["model_name", "tn", "tp", "fn", "fp", "recall", "fpr", "accuracy", "error_rate", "precision", "f_measure", "f_beta"])
    TN, TP, FN, FP = test_model(resnet, os.path.basename(model), device, batch_size=10, dim = dim)
    print(TN, TP, FN, FP)
    recall, fpr, accuracy, error_rate, precision, f_measure, f_beta = performance_measures(TN, TP, FN, FP)

    print(recall, fpr, accuracy, error_rate, precision, f_measure, f_beta)

    line = [os.path.basename(model), TN, TP, FN, FP, recall, fpr, accuracy, error_rate, precision, f_measure, f_beta]
    output_dataframe.loc[0] = line

    output_dataframe.to_csv(test_report_dir + os.path.splitext(os.path.basename(model))[0] + "_test_report.csv", index = False)