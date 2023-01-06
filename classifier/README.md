# What does this code do ?

This code is a pytorch implementation of different UNet architectures. It's supposed to be an easy to use package for training binary or semantic segmentation models and prediction segmentation masks. It is currently optimized for the segmenting C. Elegans microscopy images without body fluorescence. With little efforts, it could be tweeked to work for any kind of images. 

# How to use this code

## Prerequisites

- Split your data in two different sets (Training and Validation), one will be used for training your network's parameters, the other one will be used to test the performance of your network as it is training (so you can know when to stop the training and avoid overfitting).
- Make sure you've installed all the required libraries. For this, run the command : **pip3 install -r requirements.txt**

## Prediction

### For IZB Members, and other SLURM users

It is very simple to use, it's exactly like running matlab scripts.

- Change the file paths in the **prediction.sh** file.
- Change the model path to the path of the model you want to use.
- You can adjust the batch size by modifying the **-b** argument.
- You can ask for more time, more GPUs, more cores, as usual.
- Run the command **sbatch prediction.sh**

### For other users

- Change the file paths in the **prediction.sh** file.
- Change the model path to the path of the model you want to use.
- You can adjust the batch size by modifying the **-b** argument.
- Run the command **./prediction.sh**

Or simply run the command **python3 prediction.py [args]**

Arguments are :

- **-b** or **--batch-size** : Batch size for the prediction. Defaults to 6.
- **-i** or **--images-dir** : Path to the directory containing your input images. (REQUIRED)
- **-o** or **--output-dir** : Path to the directory where the segmentation masks will be saved. (REQUIRED)

## Training

### For IZB Members, and other SLURM users

It is also very simple to train new models.

- Change the file paths in the **train.sh** file.
- Change the model path to the path of the model you want to use.
- You can tweek the learning rate if you want but I would recommand not to touch it except if your network doesn't learn or if you're sure of what you're doing.
- If you want to switch between semantic and binary segmentation, change the METHOD argument. **CAUTION**: if you change it to "semantic", you will need to add **-c [number of classes]** at the end of the python3 line.
- You can adjust the batch size by modifying the **-b** argument.
- You can adjust the number of training epochs by modifying the **-e** argument.
- You can adjust the saving frequency (frequency at which the model will be tested on the validation set and saved, in epochs) by modifying the **-s** argument.
- You can adjust the dimension the images will be downscaled to by modifying the **-d** argument. **CAUTION**: the dimension has to be a multiple of **16** for the code to work.
- You can ask for more time, more GPUs, more cores, as usual.
- Run the command **sbatch prediction.sh**

If you want to load a pretrained model, use the **-f** or **--load** argument followed by the model's path.

### For other users

Simply run the command **python3 prediction.py [args]**

Arguments are :

- **-e** or **--epochs** : Number of training epochs. Defaults to 21.
- **-b** or **--batch-size** : Batch size for the prediction. Defaults to 1.
- **-l** or **--learning-rate** : Learning rate. Defaults to 1e-4.
- **-f** or **--load** : Load model from a .pth file.
- **-d** or **--dim** : Downscaling factor of the images. Defaults to 512. **CAUTION**: the dimension has to be a multiple of **16** for the code to work.
- **-m** or **--method** : Segmentation method, either "semantic" or "binary". Defaults to "binary". **CAUTION**: if you change it to "semantic", you will need to add the **-c [number of classes]** argument.
- **-c** or **--classes** : Number of classes for the segmentation. REQUIRED if method is set to "semantic".
- **-t** or **--save-frequency** : Save and test frequency in epochs. Defaults to 1.
- **--training-dir** : Path to the directory containing the training set. (REQUIRED)
- **--validation-dir** : Path to the directory containing the validation set. (REQUIRED)
