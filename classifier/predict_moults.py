import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import argparse
import matplotlib
import glob
import skimage.io as skio
import xgboost as xgb
import os
import shutil
import multiprocessing as mp
from itertools import repeat
import tifffile as ti
import csv
from typing import Tuple

import torch
from collections import OrderedDict


from model import ResNet50, ResNet101, ResNet152
from utils.dataset import *
from utils.array import *

from utils.dataset import PredictionDatasetList
from torch.utils.data import DataLoader
from tqdm import tqdm

from scipy import signal
import csv
import pandas as pd

labels = ["moulting", "not_moulting"]

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

def list_images_of_point(images, point):
    """
    Given a list of images and a point, returns a list of image names that contain the point in their file names.
    
    Parameters:
        images (list): List of image file names.
        point (str): Point to search for in image file names.
    
    Returns:
        list: List of image names that contain the point in their file names.
    """
    
    # Initialize empty list to store image names
    image_names = []
    
    # Iterate through list of images
    for image in images:
        # Check if point is in the image file name
        if point in os.path.basename(image):
            # If point is found, append image name to list
            image_names.append(image)
    
    # Return list of image names
    return image_names

def find_consecutive(input_list, min_consecutive_length):
    """
    Finds the consecutive elements in a list that have a length greater than or equal to min_consecutive_length.
    """
    consecutives = []
    for i, e in enumerate(input_list):
        if i == 0 or input_list[i - 1] != (e - 1):
            j = i + 1
            consecutive = [e]

            while j < (len(input_list)) and input_list[j] == (input_list[j-1] + 1):
                consecutive.append(input_list[j])
                j += 1
            if len(consecutive) >= min_consecutive_length:
                consecutives.append(consecutive)
    return consecutives

def beg_and_end(lst):
    """
    Returns the first and last element of a given list.
    
    Parameters:
    lst (list): The list for which the first and last element are to be returned.
    
    Returns:
    tuple: A tuple containing the first and last element of the list.
    
    Raises:
    TypeError: If the input is not a list.
    ValueError: If the input list is empty.
    """
    if not isinstance(lst, list):
        raise TypeError("Input must be a list.")
    if not lst:
        raise ValueError("Input list must not be empty.")
    return lst[0], lst[-1]

def get_point_number(point):
    """
    Extracts the point number from the point string.
    """
    return point[5:].lstrip("0") or "0"

def find_moults(trajectory, moult_value=1, min_width=8):
    """
    Finds the potential moults in a given trajectory based on the specified moult value and minimum width (corresponding to the minimum duration of the moult in timepoints).
    
    Parameters:
        trajectory (ndarray): The trajectory data to search for moults.
        moult_value (int, optional): The value in the trajectory data that indicates a potential moult. Defaults to 1.
        min_width (int, optional): The minimum number of consecutive points with the moult value required for it to be considered a moult. Defaults to 8.
    
    Returns:
        list: A list of lists containing the indices of the start and end of each potential moult.
    """
    if min_width < 2:
        raise ValueError("min_width must be at least 2.")
    
    potential_moults = np.argwhere(trajectory == moult_value).squeeze().tolist()

    consecutives = find_consecutive(potential_moults, min_width)
    consecutives = sorted(consecutives, key=lambda x: x[0], reverse=False)

    if len(consecutives) <= 4:
        return consecutives
    else:
        return consecutives[0:4]

def predict(images_dir: str, output_csv:str, model_path: str, batch_size: int, device: torch.device) -> None:
    """
    Makes predictions for the images in the given directory using the model at the given path.
    
    Parameters:
        images_dir (str): The directory containing the input images.
        output_csv (str) : The path to the csv file while the moults will be saved.
        model_path (str): The path to the model.
        batch_size (int): The batch size to use for prediction.
        device (torch.device): The device to run the prediction on.
    """

    images = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]

    image_titles = [os.path.basename(x) for x in images]

    points = [x.split("_")[1] for x in image_titles]
    unique_points = np.unique(points).tolist()

    point_list = [x for x in unique_points]

    resnet, epoch, dim, n_channels, classes, accuracy_val, network_name = load_model("./checkpoints/very_good_for_moults.pth", device, torch.cuda.device_count() > 1)

    logging.info(f'''Loading model:
        Model: {args.load}
        Type: {network_name}
        Epochs : {epoch}
        Accuracy on validation set : {accuracy_val}
        Classes:   {classes}
        Images dimension:  {dim}
        Device:          {device.type}
    ''')
    output_dataframe = pd.DataFrame(columns=["Point", "beg_moult1", "end_moult1", "beg_moult2", "end_moult2", "beg_moult3", "end_moult3", "beg_moult4", "end_moult4"])

    for point in point_list:

        videos_of_point = sorted(list_images_of_point(images, point))

        dataset_pred = PredictionDatasetList(videos_of_point, classes, dim)
        n_pred = len(dataset_pred)
        pred_loader = DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

        with torch.no_grad():
            trajectory = []
            with tqdm(total=n_pred, desc=f'Prediction of {point}', unit='img') as pbar:
                for batch in pred_loader:
                    imgs = batch['image']
                    img_paths = batch['img_path']

                    imgs = imgs.to(device=device, dtype=torch.float32)

                    probs = torch.softmax(resnet(imgs), dim=1)

                    del imgs
                    if (len(probs.shape) == 1):
                        probs = probs.unsqueeze(1)

                    for i in range(probs.shape[0]):
                        pred = probs[i, :]
                        max_idx = torch.argmax(pred, keepdim=True)
                        label = labels[max_idx]

                        if label == "moulting":
                            trajectory.append(1)
                        elif label == "not_moulting":
                            trajectory.append(0)
                    
                    del probs
                    torch.cuda.empty_cache()
                    pbar.update(batch_size)

        clean_traj = signal.medfilt(trajectory, 3)
        # clean_traj = trajectory

        point_line = [get_point_number(point)]
        moults = find_moults(clean_traj)
        beg_end_moults = []
        for l in moults:
            beg_end_moults.append(beg_and_end(l))

        while len(beg_end_moults) < 4:
            beg_end_moults.append((np.nan, np.nan))
        for m in beg_end_moults:
            point_line.extend([m[0], m[1]])

        print(point_line)

        output_dataframe.loc[len(output_dataframe)] = point_line

    print(output_dataframe)
    output_dataframe.to_csv(output_csv, index=False)


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
                        help='Directory containing the input images', dest='images_dir')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory', dest='output_dir')
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help='Path to the model', dest='model_path')

    # Parse the arguments and return the resulting namespace object
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    batch_size = args.batch_size
    images_dir = args.images_dir
    output_dir = args.output_dir
    model_path = args.model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_csv = output_dir + "moults.csv"

    # Run prediction
    predict(images_dir, output_csv, model_path, batch_size, device)
