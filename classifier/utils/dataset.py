import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import os
import scipy.ndimage as ndi
from skimage.exposure import equalize_adapthist
import tifffile as ti
import multiprocessing as mp
import pandas as pd

class DifferentChannelNumberException(Exception):
    """Raise for when the images in the directory have a different number of channels."""

def get_number_of_channels_img(img: str) -> int:
    """Returns the number of channels in the given image.

    Parameters:
        img (str): path to the image file

    Returns:
        int: number of channels in the image
    """
    img_array = ti.imread(img)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, 0)
    return img_array.shape[0]


def get_number_of_channels_database(dir: str) -> int:
    """
    This function returns the number of channels of the images in a directory.
    If the images in the directory have different number of channels, a DifferentChannelNumberException is raised.

    Parameters:
    dir (str): The path to the directory containing the images.

    Returns:
    int: The number of channels of the images.

    Raises:
    DifferentChannelNumberException: If the images in the directory have different number of channels.
    """
    imgs = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith(".tiff")]

    with mp.Pool(64) as pool:
        nb_channels = pool.map(get_number_of_channels_img, imgs)
    if np.min(nb_channels) != np.max(nb_channels):
        raise DifferentChannelNumberException(
            "Error ! The images in the directory have different numbers of channels.")
    else:
        return int(nb_channels[0])

# DATASET FOR TRAINING
class TrainingDataset(Dataset):
    """A PyTorch dataset for training a classifying model.

    Args:
        imgs_dir (str): Path to the directory containing the training images.
        images_labels (str): Path to .csv file containing the images' labels.
        classes (list, optional): List of the classes. If None, will be set to the list of unique labels in the data.
        dim (int, optional): Downscaling factor of the images. Defaults to 512.
    """
    def __init__(self, imgs_dir, image_labels, classes = None, dim = 512):
        self.imgs_dir = imgs_dir
        self.image_labels = pd.read_csv(image_labels)

        # Get the number of channels in the images in the directory
        self.n_channels = get_number_of_channels_database(imgs_dir)

        if classes is None:
            classes = []
            for row in self.image_labels['label']:
                if row not in classes:
                    classes.append(row)

            self.classes = sorted(classes)
        else:
            self.classes = classes

        y = np.arange(len(self.classes))
        self.one_hot_classes = np.eye(len(self.classes), dtype='uint8')[y]

        self.dim = dim
        
        self.ids = [x for x in os.listdir(
                imgs_dir) if (not x.startswith('.') and not x.endswith('.csv'))]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img, dim, n_channels, augment_contrast=True):
        """
        Preprocesses the image to be fed into the model. Optionally augments the contrast of the image.
        Scales the image down to the specified dimensions.

        Parameters:
            img (np.ndarray): The image to be preprocessed.
            dim (int): The target dimension for the image.
            augment_contrast (bool): If True, increases the contrast of the image. Defauls to True.

        Returns:
            np.ndarray: The preprocessed image.
        """
        if n_channels == 2:
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0005, kernel_size=int(img[0, :, :].shape[0]/16))
                improved_img[1, :, :] = equalize_adapthist(img[1, :, :], nbins=np.max(
                    img[1, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[1, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
                normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

        elif n_channels == 1:
            img = np.expand_dims(img, 0)
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0004, kernel_size=int(img[0, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img
        
    def __getitem__(self, i):
       
        idx = self.ids[i]
        img_file = self.imgs_dir + idx

        img_name = os.path.basename(img_file)

        img_label = self.image_labels[self.image_labels['img_id']==img_name]['label']

        assert img_label.shape[0] == 1, \
            f'Either no label or multiple labels found of the ID {idx}: {img_file}'

        image_label = img_label.iloc[0]

        img = ti.imread(img_file)
        
        img = self.preprocess_img(img, self.dim, self.n_channels)
        lbl = self.one_hot_classes[self.classes.index(image_label)]

        return {'image': torch.tensor(img), 'label': torch.tensor(lbl)}



# DATASET FROM PREDICTION

class PredictionDataset(Dataset):
    """A PyTorch dataset for classifying images.

    Args:
        imgs_dir (str): Path to the directory containing the images.
        classes (list): List of the classes.
        dim (int, optional): Downscaling factor of the images. Defaults to 512.
    """
    def __init__(self, imgs_dir, classes, dim=512):
        self.imgs_dir = imgs_dir
        self.n_channels = get_number_of_channels_database(imgs_dir)
        self.dim = dim

        self.ids = [x for x in os.listdir(
                imgs_dir) if (not x.startswith('.') and not x.endswith('.csv'))]

        self.classes = classes
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img, dim, n_channels, augment_contrast=True):
        """
        Preprocesses the image to be fed into the model. Optionally augments the contrast of the image.
        Scales the image down to the specified dimensions.

        Parameters:
            img (np.ndarray): The image to be preprocessed.
            dim (int): The target dimension for the image.
            augment_contrast (bool): If True, increases the contrast of the image. Defauls to True.

        Returns:
            np.ndarray: The preprocessed image.
        """
        if n_channels == 2:
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0005, kernel_size=int(img[0, :, :].shape[0]/16))
                improved_img[1, :, :] = equalize_adapthist(img[1, :, :], nbins=np.max(
                    img[1, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[1, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
                normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

        elif n_channels == 1:
            img = np.expand_dims(img, 0)
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0004, kernel_size=int(img[0, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

    def __getitem__(self, i):

        idx = self.ids[i]

        img_file = self.imgs_dir + idx

        img = ti.imread(img_file)

        img = self.preprocess_img(img, self.dim, self.n_channels)

        return {'image': torch.tensor(img), 'img_path': img_file}

class PredictionDatasetList(Dataset):
    """A PyTorch dataset for classifying a list of images.

    Args:
        list_img (list): List of images to be classified.
        classes (list): List of the classes.
        dim (int, optional): Downscaling factor of the images. Defaults to 512.
    """
    def __init__(self, list_img, classes, dim=512):
        self.n_channels = get_number_of_channels_database(list_img)
        self.classes = classes

        self.dim = dim

        self.ids = [x for x in list_img if (not x.startswith('.') and not x.endswith('.csv'))]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img, dim, n_channels, augment_contrast=True):
        """
        Preprocesses the image to be fed into the model. Optionally augments the contrast of the image.
        Scales the image down to the specified dimensions.

        Parameters:
            img (np.ndarray): The image to be preprocessed.
            dim (int): The target dimension for the image.
            augment_contrast (bool): If True, increases the contrast of the image. Defauls to True.

        Returns:
            np.ndarray: The preprocessed image.
        """
        if n_channels == 2:
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0005, kernel_size=int(img[0, :, :].shape[0]/16))
                improved_img[1, :, :] = equalize_adapthist(img[1, :, :], nbins=np.max(
                    img[1, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[1, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
                normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

        elif n_channels == 1:
            img = np.expand_dims(img, 0)
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0004, kernel_size=int(img[0, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

    def __getitem__(self, i):

        idx = self.ids[i]

        img_file = idx

        img = ti.imread(idx)

        img = self.preprocess_img_or_msk(img, self.dim)

        return {'image': torch.tensor(img), 'img_path': img_file}