# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
"""
    File Name: randomaug.py - 
    Description: This file contains various image augmentation functions that can be used to enhance the diversity of training data for machine learning models. The augmentations include geometric transformations (shearing, translation, rotation), color adjustments (contrast, brightness, sharpness), and other effects (solarization, posterization, cutout). Additionally, it includes a RandAugment class that applies a random selection of these augmentations to an image.

    Required Libraries: 
        PIL - using for image processing
        numpy - for numerical operations
        torch - for tensor operations

        
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

    Created by: Andrew Iorio
 """

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(img, v):  # [-0.3, 0.3]
    """
        shearX augmentation, shears the image along the X axis with a factor of v

        Args:
            img (PIL.Image): The input image to be sheared.
            v (float): The shear factor, should be in the range [-0.3, 0.3].
        Returns:
            PIL.Image: The sheared image.
    """
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    """
        ShearY augmentation, shears the image along the Y axis with a factor of v

        Args:
            img (PIL.Image): The input image to be sheared.
            v (float): The shear factor, should be in the range [-0.3, 0.3].
        Returns:
            PIL.Image: The sheared image.
    """
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    Docstring for TranslateX
    
    Args:
        img (PIL.Image): The input image to be translated.
        v (float): The translation factor, should be in the range [-0.45, 0.45].

    Returns:
        PIL.Image: The translated image.

    """
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    TranslateXabs - Translates the image along the X axis by an absolute value v.
    
    Args:
        img (PIL.Image): The input image to be translated.
        v (float): The absolute translation value, should be non-negative.

    Returns:
        PIL.Image: The translated image.

    """
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    TranslateY - Translates the image along the Y axis by a factor of v.

    Args:
        img (PIL.Image): The input image to be translated.
        v (float): The translation factor, should be in the range [-0.45, 0.45].

    Returns:
        PIL.Image: The translated image.

    """
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    """
    Docstring for TranslateYabs
    
    Args:
        img (PIL.Image): The input image to be translated.
        v (float): The absolute translation value, should be non-negative.

    Returns:
        PIL.Image: The translated image.
    """
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    """
    Rotate - Rotates the image by an angle v.

    Args:
        img (PIL.Image): The input image to be rotated.
        v (float): The rotation angle in degrees, should be in the range [-30, 30].

    Returns:
        PIL.Image: The rotated image.

    """
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    """
    AutoContrast - Applies auto-contrast to the image.

    Args:
        img (PIL.Image): The input image to be auto-contrasted.

    Returns:
        PIL.Image: The auto-contrasted image.
    """
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    """
    Invert - Inverts the colors of the image.

    Args:
        img (PIL.Image): The input image to be inverted.

    Returns:
        PIL.Image: The inverted image.
    """

    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    """
    Equalize - Equalizes the histogram of the image.

    Args:
        img (PIL.Image): The input image to be equalized.

    Returns:
        PIL.Image: The equalized image.
    """
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    """
    Flip - Flips the image horizontally.

    Args:
        img (PIL.Image): The input image to be flipped.
        _ (any): Placeholder parameter.

    Returns:
        PIL.Image: The flipped image.
    """
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    """
    Solarize - Applies solarization to the image.

    Args:
        img (PIL.Image): The input image to be solarized.
        v (int): The solarization threshold, should be in the range [0, 256].

    Returns:
        PIL.Image: The solarized image.
    """
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    """
    SolarizeAdd - Applies solarization to the image with an addition.

    Args:
        img (PIL.Image): The input image to be solarized.
        addition (int): The value to add to the image.
        threshold (int): The solarization threshold, should be in the range [0, 256].

    Returns:
        PIL.Image: The solarized image.
    """
    img_np = np.array(img).astype(int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    """
    Posterize - Applies posterization to the image.

    Args:
        img (PIL.Image): The input image to be posterized.
        v (int): The number of bits to keep, should be in the range [4, 8].

    Returns:
        PIL.Image: The posterized image.
    """
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    """
        Contrast - Adjusts the contrast of the image.
        
        Args:
            img (PIL.Image): The input image to be adjusted.
            v (float): The contrast factor, should be in the range [0.1, 1.9].
            Returns:
            PIL.Image: The contrast-adjusted image.
    """
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    """
        Color - Adjusts the color of the image.

        Args:
            img (PIL.Image): The input image to be adjusted.
            v (float): The color factor, should be in the range [0.1, 1.9].

        Returns:
            PIL.Image: The color-adjusted image.
    """

    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    """
    Brightness - Adjusts the brightness of the image.
    
    Args:
        img (PIL.Image): The input image to be adjusted.
        v (float): The brightness factor, should be in the range [0.1, 1.9].

    Returns:
        PIL.Image: The brightness-adjusted image.

    """ 
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    """
    Sharpness - Adjusts the sharpness of the image.

    Args:
        img (PIL.Image): The input image to be adjusted.
        v (float): The sharpness factor, should be in the range [0.1, 1.9].

    Returns:
        PIL.Image: The sharpness-adjusted image.
    """ 
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    """
    Cutout - Applies cutout to the image.

    Args:
        img (PIL.Image): The input image to be cutout.

        v (float): The cutout factor, should be in the range [0.0, 0.2].

    Returns:
        PIL.Image: The cutout image.

    """
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    """
    CutoutAbs - Applies cutout to the image.

    Args:
        img (PIL.Image): The input image to be cutout.
        v (float): The cutout factor, should be in the range [0.0, 0.2].

    Returns:
        PIL.Image: The cutout image.
    """
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    """
    SamplePairing - Samples a pair of images.

    Args:
        imgs (list): List of images to sample from.

    Returns:
        function: A function that blends two images.
    """
    def f(img1, v):
        """
        f - Blends two images.
        
        inner function to blend two images.

        Args:
            img1 (PIL.Image): The first input image.
            v (float): The blending factor.
        Returns:
            PIL.Image: The blended image.

        """
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    """
    Identity - Returns the image unchanged.
    
    Args:
        img (PIL.Image): The input image.
        v (any): Placeholder parameter.

    Returns:
        PIL.Image: The unchanged image.
    """
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


def augment_list_64():  # 16 oeprations and their ranges
    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 20),               # ↓ from 30 to 20
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.2),             # ↓ from 0.3 to 0.2
        (ShearY, 0., 0.2),
        (CutoutAbs, 0, 16),            # ↓ from 40 to 16 (≈25% of 64)
        (TranslateXabs, 0., 10),       # ↓ from 100 to 10 px
        (TranslateYabs, 0., 10),
    ]
    return l

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)

    funtions:
        __init__(self, alphastd, eigval, eigvec): Initializes the Lighting
        __call__(self, img): Applies lighting noise to the image.

    Variables:
        alphastd (float): Standard deviation for the noise.
        eigval (torch.Tensor): Eigenvalues for the PCA.
        eigvec (torch.Tensor): Eigenvectors for the PCA.

    
    """

    def __init__(self, alphastd, eigval, eigvec):
        """
        Initializes the Lighting
        Args:
            alphastd (float): Standard deviation for the noise.
            eigval (list): Eigenvalues for the PCA.
            eigvec (list): Eigenvectors for the PCA.
        """
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        """
        __call_ - Applies lighting noise to the image.
        
        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The image tensor with lighting noise applied.
        """
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

def augment_list_32():
    """
    augment_list_32 - Returns a list of augmentation operations with their respective ranges for 32x32 images.

    Returns:
        list: A list of tuples containing augmentation operations and their ranges.
    """
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 15),               # Down from 30 to 15 degrees
        (Posterize, 4, 8),             # 4-bit to 8-bit
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 8),             # ~25% of image: 8 pixels
        (TranslateXabs, 0., 4),        # ~12.5% of width
        (TranslateYabs, 0., 4),        # ~12.5% of height
    ]
    return l

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py

    CutoutDefault - Applies cutout to a tensor image.
    
    funtions:
        __init__(self, length): Initializes the CutoutDefault
        __call__(self, img): Applies cutout to the image.

    Variables:
        length (int): The length of the cutout square.
    """
    def __init__(self, length):
        """
        Initializes the CutoutDefault
        Args:
            length (int): The length of the cutout square.


        """
        self.length = length

    def __call__(self, img):
        """
        __call_ - Applies cutout to the image.

        Args:
            img (torch.Tensor): The input image tensor.
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    """
    RandAugment - Applies a random selection of augmentations to an image.

    funtions:
        __init__(self, n, m, img_size): Initializes the RandAugment

        __call__(self, img): Applies n random augmentations to the image with magnitude m.

    Variables:
        n (int): Number of augmentations to apply.
        m (int): Magnitude of the augmentations.
        img_size (int): Size of the input image (32, 64, or 224).
    """
    def __init__(self, n, m, img_size):
        """
        Initializes the RandAugment
        Args:
            n (int): Number of augmentations to apply.
            m (int): Magnitude of the augmentations.
            img_size (int): Size of the input image (32, 64, or 224).

        returns:

        """
        self.n = n
        self.m = m      
        if img_size == 32:
            self.augment_list = augment_list_32()
        if img_size == 64:
            self.augment_list = augment_list_64()
        else: # for 224 sized images
            self.augment_list = augment_list()

    def __call__(self, img):
        """
        __call_ - Applies n random augmentations to the image with magnitude m.

        Args:
            img (PIL.Image): The input image to be augmented.

        Returns:
            PIL.Image: The augmented image.
        """
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img
