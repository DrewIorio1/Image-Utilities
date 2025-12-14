The following classes and functions are to provide image sample functions.

File Name randomaug.py -  This file contains various image augmentation functions that can be used to enhance the diversity of training data for machine learning models. The augmentations include geometric transformations (shearing, translation, rotation), color adjustments (contrast, brightness, sharpness), and other effects (solarization, posterization, cutout). Additionally, it includes a RandAugment class that applies a random selection of these augmentations to an image.

    Classes and Functions:
        ShearX, ShearY - Shearing transformations along X and Y axes.
        TranslateX, TranslateY - Translation transformations along X and Y axes.
        Rotate - Rotation transformation.
        AutoContrast, Invert, Equalize - Color adjustments.
        Solarize, Posterize - Image effects.
        Contrast, Color, Brightness, Sharpness - Color and brightness adjustments.
        Cutout - Cutout augmentation.
        SamplePairing - Samples a pair of images for blending.
        Identity - Returns the image unchanged.
        augment_list, augment_list_64, augment_list_32 - Functions that return lists of augmentation operations with their respective ranges.
        RandAugment - Class that applies a random selection of augmentations to an image.

    Class Lighting - Applies lighting noise based on PCA.
        __init__(self, alphastd, eigval, eigvec) - Initializes the Lighting
        __call__(self, img) - Applies lighting noise to the image.

    Class CutoutDefault - Applies cutout to a tensor image.
        __init__(self, length) - Initializes the CutoutDefault
        __call__(self, img) - Applies cutout to the image.


    Class RandAugment:
        __init__(self, n, m, img_size) - Initializes the RandAugment
        __call__(self, img) - Applies n random augmentations to the image with magnitude m.
