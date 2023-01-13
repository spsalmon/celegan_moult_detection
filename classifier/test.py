import argparse
import logging
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from model import ResNet50, ResNet101, ResNet152
from utils.dataset import *
from utils.array import *

from collections import OrderedDict
from typing import Tuple

import pandas as pd

logging.basicConfig()


def load_model(model_path: str, device: torch.device, parallel: bool) -> Tuple[nn.Module, int, int, int, list, float, str]:
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
    # n_channels = checkpoint['n_channels']
    n_channels = 1

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


def test_model(net, img_dir, labels_csv, model_name, classes, device, batch_size=1, dim=512):
    """
    Test a binary classifier model on a dataset.

    Parameters
    ----------
    net: torch.nn.Module
        The trained model to be tested.
    img_dir: str
        The directory containing the test images.
    labels_csv: str
        The file path of the CSV file containing the labels for the test images.
    model_name: str
        The name of the model.
    classes: list of str
        A list of the class labels.
    device: str
        The device to use for testing (e.g. 'cpu' or 'cuda').
    batch_size: int, optional
        The batch size to use for testing. Default is 1.
    dim: int, optional
        The size of the images in the dataset. Default is 512.

    Returns
    -------
    tuple
        A tuple of integers containing the following counts:
        - The number of true negatives (TN).
        - The number of true positives (TP).
        - The number of false negatives (FN).
        - The number of false positives (FP).
    """
    # Create a test dataset and dataloader
    dataset_test = TrainingDataset(img_dir, labels_csv, classes, dim=dim)
    n_test = len(dataset_test)
    test_loader = DataLoader(dataset_test, batch_size=batch_size,
                             shuffle=False, num_workers=32, pin_memory=True)

    # Initialize counters for true negatives, true positives, false negatives, and false positives
    TN = 0
    TP = 0
    FN = 0
    FP = 0

    # Test the model
    with torch.no_grad():
        with tqdm(total=n_test, desc=f'Testing of {model_name}', unit='img') as pbar:
            for batch in test_loader:
                # Get the images and true labels from the batch
                imgs = batch['image']
                true_labels = batch['label']

                # Move the images to the specified device
                imgs = imgs.to(device=device, dtype=torch.float32)

                # Get the model's predictions for the images
                probs = torch.softmax(net(imgs), dim=1)

                # Delete the images to save memory
                del imgs

                # If probs has only one dimension, add a second dimension of size 1
                if (len(probs.shape) == 1):
                    probs = probs.unsqueeze(1)

                # Iterate through the predictions of the batch and update the counters
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


def performance_measures(TN: int, TP: int, FN: int, FP: int) -> Tuple[float, float, float, float, float, float, float]:
    """
    Calculate various performance measures for a binary classifier.

    Parameters:
    TN: int
        The number of true negatives.
    TP: int
        The number of true positives.
    FN: int
        The number of false negatives.
    FP: int
        The number of false positives.

    Returns:
    tuple : a tuple containing the following performance measures:
        - 'recall': float
        - 'fpr': float
        - 'accuracy': float
        - 'error_rate': float
        - 'precision': float
        - 'f_measure': float
        - 'f_beta': float
    """
    # Calculate recall
    recall = TP / (TP + FN)

    # Calculate false positive rate (fpr)
    fpr = FP / (TN + FP)

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Calculate error rate
    error_rate = 1 - accuracy

    # Calculate precision
    precision = TP / (TP + FP)

    # Calculate F-measure
    f_measure = 2 * (precision * recall) / (precision + recall)

    # Calculate F-beta
    f_beta = (1 + 0.5**2) * ((precision * recall) /
                             (0.5**2 * precision + recall))

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
    model_path = args.model_path
    test_dir = args.test_dir
    test_report_dir = args.report_dir
    labels_csv = os.path.join(test_dir, "labels.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet, epoch, dim, n_channels, classes, accuracy_val, network_name = load_model(
        model_path, device, torch.cuda.device_count() > 1)

    output_dataframe = pd.DataFrame(columns=["model_name", "tn", "tp", "fn", "fp",
                                    "recall", "fpr", "accuracy", "error_rate", "precision", "f_measure", "f_beta"])
    TN, TP, FN, FP = test_model(resnet, test_dir, labels_csv, os.path.basename(model_path), classes = classes, device=device, batch_size=10, dim=dim)
    print(TN, TP, FN, FP)
    recall, fpr, accuracy, error_rate, precision, f_measure, f_beta = performance_measures(
        TN, TP, FN, FP)

    print(recall, fpr, accuracy, error_rate, precision, f_measure, f_beta)

    line = [os.path.basename(model_path), TN, TP, FN, FP, recall,
            fpr, accuracy, error_rate, precision, f_measure, f_beta]
    output_dataframe.loc[0] = line

    output_dataframe.to_csv(test_report_dir + os.path.splitext(
        os.path.basename(model_path))[0] + "_test_report.csv", index=False)
