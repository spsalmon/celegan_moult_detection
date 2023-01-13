import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.transform import resize

def preprocess_img(img, dim, n_channels, augment_contrast=True):
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

                # # Resize the image
                # scale_x = dim/img.shape[1]
                # scale_y = dim/img.shape[2]
                processed_img = resize(improved_img, (2, dim, dim))
                # processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
                normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

                # # Resize the image
                # scale_x = dim/img.shape[1]
                # scale_y = dim/img.shape[2]

                processed_img = resize(normalized_img, (2, dim, dim))
                # processed_img = ndi.zoom(normalized_img, zoom=[
                #                         1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

        elif n_channels == 1:
            img = np.expand_dims(img, 0)
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0004, kernel_size=int(img[0, :, :].shape[0]/16))

                # # Resize the image
                # scale_x = dim/img.shape[1]
                # scale_y = dim/img.shape[2]
                processed_img = resize(improved_img, (1, dim, dim))
                # processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])

                # # Resize the image
                # scale_x = dim/img.shape[1]
                # scale_y = dim/img.shape[2]

                # processed_img = ndi.zoom(normalized_img, zoom=[
                #                         1, scale_x, scale_y])
                processed_img = resize(normalized_img, (1, dim, dim))
                processed_img = processed_img.astype('float64')
            return processed_img