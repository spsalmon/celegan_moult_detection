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
from utils.loss_function import *
from utils.accuracy import IoU
from utils.dataset import *
from utils.array import *
from scipy import interpolate

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import TrainingDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image, ImageFilter
from collections import OrderedDict

from pathlib import Path
import skimage
from skimage.morphology import thin, skeletonize

import nvidia_smi

dir_img_train = "/mnt/external.data/TowbinLab/spsalmon/moult_database/fluo/training/"
labels_train = "/mnt/external.data/TowbinLab/spsalmon/moult_database/fluo/training/labels.csv"

def load_model(model_path, device, n_classes):
    net = ResNet50(n_classes, 1)
    # Load Nested Unet
    # net = NestedUNet(n_channels=2, n_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    net.to(device=device)

    state_dict = checkpoint['model_state_dict']
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    
    net.load_state_dict(new_state_dict)

    epoch = checkpoint['epoch']
    accuracy_val = checkpoint['accuracy_val']
    del checkpoint
    torch.cuda.empty_cache()
    print("Model : epoch = %s | accuracy = %s" % (epoch, accuracy_val))

    return {'net': net, 'epoch':epoch}

def load_model_parallel(model_path, device, n_classes):
    net = ResNet50(n_classes, 1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    epoch = checkpoint['epoch']
    accuracy_val = checkpoint['accuracy_val']
    del checkpoint
    torch.cuda.empty_cache()
    print("Model : epoch = %s | accuracy = %s" % (epoch, accuracy_val))

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    logging.info(f'Model loaded from {model_path}')

    del state_dict
    torch.cuda.empty_cache()

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    unet = nn.DataParallel(net)
    unet.cuda()

    return unet


def resize_pred_binary(pred, shape):
    if (len(pred.shape) == 4):
        pred = pred.squeeze(0).cpu().numpy()
    else:
        pred = pred.cpu().numpy()

    x, y, z = pred.shape[0], pred.shape[1], pred.shape[2]

    return ndimage.zoom(pred, (x, float(shape[1] / y), float(shape[2] / z)), order=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_GPU_memory():
    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(
            i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

    nvidia_smi.nvmlShutdown()


def fake_training(net,
              device,
              labels,
              epochs = 50,
              batch_size = 3,
              lr = 0.1,
              save_cp = True,
              save_frequency = 10,
              dim = 512):

    # print("###### LOADING NETWORK ######")

    # check_GPU_memory()

    # Load the dataset
    dataset = TrainingDataset(dir_img_train, labels_train, labels, dim)
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Init the optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, epochs=epochs, steps_per_epoch=len(train_loader))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, threshold=0.0005)

    # Init the loss function
    criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    # Loop for the epoch

    net.train()

    epoch_loss, epoch_step, epoch_loss_val = 0, 0, 0
    correct_train, element_train, correct_organ, element_organ = 0, 0, 0, 0
    accuracy_val, loss_val, IoU_val = 0, 0, 0

    dataloader_iterator = iter(train_loader)

    # print("###### TRAINING BEGINS ######")
    # check_GPU_memory()

    for i in range(30):
        print("batch", i)
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_loader)
            batch = next(dataloader_iterator)

        print("###### LOADING IMAGES ######")

        imgs = batch['image']
        true_labels = batch['label']

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_labels = true_labels.to(device=device, dtype = torch.float32)

        check_GPU_memory()

        print("###### PREDICTION ######")

        probs = net(imgs)

        del imgs

        check_GPU_memory()

        print("###### BACKPROPAGATION ######")

        # Compute loss function and prediction
        loss = criterion(probs, true_labels)

        del probs

        epoch_loss += float(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        check_GPU_memory()

        print("###### CLEANING ######")

        del loss

        torch.cuda.empty_cache()

        check_GPU_memory()


def fake_pred(net,
              device,
              method,
              epochs=50,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              save_frequency=10,
              dim=512):

    # print("###### LOADING NETWORK ######")

    # check_GPU_memory()

    dataset_pred = PredictionDataset(dir_img_train, 2, 1, dim=dim)
    pred_loader = DataLoader(dataset_pred, batch_size=batch_size,
                             shuffle=False, num_workers=32, pin_memory=True)

    with torch.no_grad():
        dataloader_iterator = iter(pred_loader)

        for i in range(30):
            print("batch", i)
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(pred_loader)
                batch = next(dataloader_iterator)

            print("###### LOADING IMAGES ######")

            imgs = batch['image']
            img_paths = batch['img_path'][0]

            imgs = imgs.to(device=device, dtype=torch.float32)

            check_GPU_memory()

            print("###### PREDICTION ######")

            probs = torch.softmax(net(imgs), dim=1)

            del imgs
            if (len(probs.shape) == 1):
                probs = probs.unsqueeze(1)

            for i in range(probs.shape[0]):
                pred = probs[i, :]
                max_idx = torch.argmax(pred, keepdim=True)
                one_hot_pred = torch.FloatTensor(pred.shape).to(
                    device=device, dtype=torch.float32)
                one_hot_pred.zero_()
                one_hot_pred.scatter_(0, max_idx, 1)

            del probs

            check_GPU_memory()

            print("###### CLEANING ######")
            torch.cuda.empty_cache()
            check_GPU_memory()


net = ResNet50(2, channels=1)

labels = ["moulting", "not_moulting"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    resnet = load_model_parallel("./checkpoints/CP_epoch25.pth", device, 2)
else:
    resnet = load_model("./checkpoints/CP_epoch25.pth", device, 2)

resnet.to(device=device)
torch.cuda.empty_cache()
fake_pred(resnet, device, labels, batch_size=26, dim=1184)
