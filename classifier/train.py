import argparse
import logging
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model import ResNet50, ResNet101, ResNet152
from utils.dataset import *
from utils.array import *

from torch.utils.tensorboard.writer import SummaryWriter
from utils.dataset import TrainingDataset
from torch.utils.data import DataLoader
from collections import OrderedDict

def load_model_parallel(model_path: str, device: torch.device) -> Tuple[nn.Module, int, int, int, list, float, str]:
    """
    Loads the model from the given path and returns it as a DataParallel model, along with other relevant information.

    Parameters:
        model_path (str): The path to the model checkpoint.
        device (torch.device): The device to load the model onto.

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

    # Wrap the model in a DataParallel module
    net = nn.parallel.DataParallel(net)

    # Move model to the chosen device
    net.to(device=device)

    # Return network and relevant information about it
    return net, epoch, dim, n_channels, classes, accuracy_val, network_name

def train_net(net,
              classes,
              device,
              dataset_train,
              dataset_val,
              layers,
              epochs=50,
              batch_size=1,
              lr=1e-4,
              save_cp=True,
              dir_checkpoint="./checkpoints",
              save_frequency=10,
              dim=512):

    # Load the datasets

    n_train = len(dataset_train)
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

    n_val = len(dataset_val)
    val_loader = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=True, num_workers=32, pin_memory=True)

    # Init the tensorboard
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_DIM_{dim}')
    global_step = 0

    # Print the information
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images dim:  {dim}
        Classes:          {classes}
    ''')

    # Print the information of the graphic card
    if device.type == 'cuda':
        print('Number of gpu : ', torch.cuda.device_count())
        print('Working on : ', torch.cuda.current_device())
        print('Name of the working gpu : ',
              torch.cuda.get_device_name(torch.cuda.current_device()))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    # Init the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))

    criterion = nn.BCEWithLogitsLoss()

    # Loop for the epoch
    for epoch in range(epochs):
        net.train()

        epoch_loss, epoch_step, epoch_loss_val = 0, 0, 0
        accuracy_val, loss_val = 0, 0

        # Init loop for the step
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                global_step += 1
                epoch_step += 1

                imgs = batch['image']
                true_labels = batch['label']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_labels = true_labels.to(
                    device=device, dtype=torch.float32)
                probs = net(imgs)

                del imgs
                # Compute loss function and prediction
                loss = criterion(probs, true_labels)

                del probs
                del true_labels
                epoch_loss += float(loss.item())
                writer.add_scalar('Loss/train/global',
                                  loss.item(), global_step)
                writer.add_scalar('Loss/train/' + str(epoch+1),
                                  loss.item(), epoch_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

                pbar.update(batch_size)

                del loss
                torch.cuda.empty_cache()
        print("EPOCH", epoch, "AVERAGE TRAINING LOSS :",
              (epoch_loss/(n_train/batch_size)))

        # This controls how often you're saving a checkpoint and testing your network on the
        # validation set.
        if epoch % save_frequency == 0:
            net.eval()
            with torch.no_grad():
                with tqdm(total=n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                    for batch in val_loader:
                        imgs = batch['image']
                        true_labels = batch['label']

                        imgs = imgs.to(device=device, dtype=torch.float32)
                        true_labels = true_labels.to(
                            device=device, dtype=torch.float32)

                        probs = net(imgs)


                        loss = criterion(probs, true_labels)

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
                            if (torch.equal(one_hot_pred, true_labels[i, :])):
                                accuracy_val += 1

                        epoch_loss_val += float(loss.item())

                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        pbar.update(batch_size)
                        del loss
                        del probs

                        torch.cuda.empty_cache()

            loss_val = (epoch_loss_val/(n_val/batch_size))
            accuracy_val = accuracy_val/n_val
            print("EPOCH", epoch, "AVERAGE LOSS :", loss_val,
                  "ACCURACY :", accuracy_val*100, "%")
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'epoch': epoch + 1,
                            'layers': layers,
                            'dim': dim,
                            'classes':classes,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss_val': loss_val,
                            'accuracy_val': accuracy_val,
                            'loss_train': (epoch_loss/(n_train/batch_size))}, dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add the arguments to the parser
    parser.add_argument('-e', '--epochs', dest="epochs", type=int, default=21,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch-size', dest="batch_size", type=int, default=1,
                        help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest="lr", type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--dim', dest='dim', type=int, default=512,
                        help='Downscaling factor of the images')
    parser.add_argument('-c', '--classes', dest='classes', type=str, default=None,
                        help='List of the classes you want to classify your data into.')
    parser.add_argument('-t', '--save-frequency', dest='save_frequency', type=int, default=1,
                        help='Save and test frequency')
    parser.add_argument('--layers', dest='layers', type=int, default=50,
                        choices=[50, 101, 152],
                        help='Number of convolutional layers, should be either 50, 101 or 152.')
    parser.add_argument('--training-dir', dest='training_dir', type=str, required=True,
                        help='Path to the directory containing the training set')
    parser.add_argument('--validation-dir', dest='validation_dir', type=str, required=True,
                        help='Path to the directory containing the validation set')
    parser.add_argument('--checkpoint-dir', dest='dir_checkpoint', type=str, default="./checkpoints/",
                        help='Path to the directory where the model checkpoints will be saved')

    # Parse the arguments and return the resulting namespace object
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    classes = args.classes
    dim = args.dim
    layers = args.layers
    n_channels = None
    net = ResNet50(1,1)

    if classes is not None:
        classes = list(map(str, classes.strip('[]').split(',')))
        for i, c in enumerate(classes):
            classes[i] = c.strip()

    if args.load:
        del net
        net, epoch, dim, n_channels, classes, accuracy_val, network_name= load_model_parallel(args.load, device)

        logging.info(f'''Loading model:
        Model: {args.load}
        Type: {network_name}
        Epochs : {epoch}
        Accuracy on validation set : {accuracy_val}
        Classes:   {classes}
        Images dimension:  {dim}
        Device:          {device.type}
    ''')

    # Creating training and validation datasets
    dir_img_train = args.training_dir
    labels_train = dir_img_train + "labels.csv"

    dir_img_val = args.validation_dir
    labels_val = dir_img_val + "labels.csv"

    logging.info(f'Loading training dataset')
    dataset_train = TrainingDataset(
        dir_img_train, labels_train, classes=classes, dim=dim)
    logging.info(f'Finished loading training dataset')

    logging.info(f'Loading validation dataset')
    dataset_val = TrainingDataset(
        dir_img_val, labels_val, classes=classes, dim=dim)
    logging.info(f'Finished loading validation dataset')

    assert dataset_train.n_channels == dataset_val.n_channels, "Training and validation dataset images don't have the same number of channels."

    if args.load:
        assert n_channels == dataset_train.n_channels, f"The model you're loading was trained on images with {n_channels} channels, and the images in your database have {dataset_train.n_channels} channels"
    else:
        del net
        # Initializing the neural network
        if layers == 50:
            network_name = "ResNet50"
            net = ResNet50(len(classes), dataset_train.n_channels)
        elif layers == 101:
            network_name = "ResNet101"
            net = ResNet101(len(classes), dataset_train.n_channels)
        elif layers == 152:
            network_name = "ResNet152"
            net = ResNet152(len(classes), dataset_train.n_channels)
        else:
            raise ValueError("The number of layers can only be 50, 101 or 152.")

        logging.info(f'Network: {network_name}\n'
                    f'\t{n_channels} input channels\n'
                    f'\t{len(classes)} output channels (classes)\n'
                    f'\t Classes : {classes}\n')
        net = nn.parallel.DataParallel(net)
    net.to(device=device)

    try:
        train_net(net=net,
                  classes=classes,
                  device=device,
                  epochs=args.epochs,
                  dataset_train=dataset_train,
                  dataset_val=dataset_val,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  dir_checkpoint=args.dir_checkpoint,
                  save_frequency=args.save_frequency,
                  dim=args.dim,
                  layers=layers)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)