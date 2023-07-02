# Required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import time
import os
import sys
import pydicom
import nrrd
import scipy.ndimage
import scipy.misc
import pickle
import random
import skimage
from collections import Counter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter




def rotate(voxels, lbls, theta=None):
    # Rotate volume by a minor angle (+/- 10 degrees: determined by investigation of dataset variability)
    if theta is None:
        theta = random.randint(-10, 10)
    vox_new = scipy.ndimage.interpolation.rotate(voxels, theta, reshape=False)
    lbl_new = scipy.ndimage.interpolation.rotate(lbls, theta, reshape=False)
    return vox_new, lbl_new


def scale_and_crop(voxels, lbls):
    # Scale the volume by a minor size and crop around centre (can also modify for random crop)
    o_s = voxels.shape
    r_s = [0] * len(o_s)
    scale_factor = random.uniform(1, 1.2)
    vox_zoom = scipy.ndimage.interpolation.zoom(voxels, scale_factor, order=1)
    lbl_zoom = scipy.ndimage.interpolation.zoom(lbls, scale_factor, order=0)
    new_shape = vox_zoom.shape
    # Start with offset
    for i in range(len(o_s)):
        if new_shape[i] == 1:
            r_s[i] = 0
            continue
        r_c = int(((new_shape[i] - o_s[i]) - 1) / 2)
        r_s[i] = r_c
    r_e = [r_s[i] + o_s[i] for i in list(range(len(o_s)))]
    vox_zoom = vox_zoom[r_s[0]:r_e[0], r_s[1]:r_e[1], r_s[2]:r_e[2]]
    lbl_zoom = lbl_zoom[r_s[0]:r_e[0], r_s[1]:r_e[1], r_s[2]:r_e[2]]
    return vox_zoom, lbl_zoom


def grayscale_variation(voxels, lbls):
    # Introduce a random global increment in gray-level value of volume.
    im_min = np.min(voxels)
    im_max = np.max(voxels)
    mean = np.random.normal(0, 0.1) # original (0, 0.1)
    smp = np.random.normal(mean, 0.01, size=np.shape(voxels)) # original 0.01
    voxels = voxels + im_max * smp
    voxels[voxels <= im_min] = im_min  # Clamp to min value
    voxels[voxels > im_max] = im_max  # Clamp to max value
    return voxels, lbls


def elastic_deformation(voxels, lbls, alpha=None, sigma=None, mode="constant", cval=0, is_random=False):
    # Apply elastic deformation/distortion to the wolume
    # Adapted from: https://tensorlayer.readthedocs.io/en/stable/_modules/tensorlayer/prepro.html#elastic_transform
    if alpha == None:
        alpha = voxels.shape[1] * 3.
    if sigma == None:
        sigma = voxels.shape[1] * 0.07
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))

    if len(voxels.shape) == 3:
        voxels = np.reshape(voxels, (voxels.shape[0], voxels.shape[1], voxels.shape[2], 1))
        lbls = np.reshape(lbls, (lbls.shape[0], lbls.shape[1], lbls.shape[2], 1))

    shape = (voxels.shape[0], voxels.shape[1])
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))

    new_voxels = np.zeros(voxels.shape)
    new_lbls = np.zeros(lbls.shape)
    for i in range(voxels.shape[2]):  # apply the same distortion to each slice within the volume
        new_voxels[:, :, i, 0] = map_coordinates(voxels[:, :, i, 0], indices, order=1).reshape(shape)
        new_lbls[:, :, i, 0] = map_coordinates(lbls[:, :, i, 0], indices, order=1).reshape(shape)

    return new_voxels, new_lbls


def intensity_shift(voxels, lbls, shift_range=None):
    # Shift the intensity values of the volume by a random amount within the specified range
    if shift_range is None:
        shift_range = [3, 5]  # Default range for intensity shift

    min_intensity = np.min(voxels)
    max_intensity = np.max(voxels)

    # Generate a random shift factor within the specified range
    shift_factor = np.random.uniform(shift_range[0], shift_range[1])

    # Apply the intensity shift to the volume
    vox_shifted = voxels * shift_factor

    # Clamp the intensity values to the original intensity range
    vox_shifted[vox_shifted < min_intensity] = min_intensity
    vox_shifted[vox_shifted > max_intensity] = max_intensity

    return vox_shifted, lbls

def sample_with_p(p):
    # Helper function to return boolean of a sample with given probability p
    if random.random() < p:
        return True
    else:
        return False


def get_random_perturbation(voxels, lbls):
    # Generate a random perturbation of the input feature + label
    # original p - 0.6 for all
    p_rotate = 0.6
    p_scale = 0.6
    p_gray = 0.6
    p_deform = 0.6
    p_intense = 0.6
    new_voxels, new_lbls = voxels, lbls
    if sample_with_p(p_rotate):
        new_voxels, new_lbls = rotate(new_voxels, new_lbls)
    if sample_with_p(p_scale):
        new_voxels, new_lbls = scale_and_crop(new_voxels, new_lbls)
    if sample_with_p(p_gray):
        new_voxels, new_lbls = grayscale_variation(new_voxels, new_lbls)
    if sample_with_p(p_deform):
        new_voxels, new_lbls = elastic_deformation(new_voxels, new_lbls)
    # new augmentation - intensity shift
    if sample_with_p(p_intense):
        new_voxels, new_lbls = intensity_shift(new_voxels, new_lbls)
    return new_voxels, new_lbls


def plot_augmentation(img_org, ann_org, img_aug, ann_aug, title_aug='Augmented', axis_off=True):
    # Plot original and augmented image along with its annotation
    # Works for different kinds of augmentations
    if len(ann_org) == 0:
        n = 1  # Annotations are not plotted
    else:
        n = 2  # Annotations are plotted
    plt.figure(figsize=(5, 5))
    plt.subplot(n, 2, 1)
    if axis_off:
        plt.axis('off')
    plt.title("Original scan")
    plt.imshow(img_org, cmap=plt.cm.gray)
    plt.subplot(n, 2, 2)
    if axis_off:
        plt.axis('off')
    plt.title(title_aug)
    plt.imshow(img_aug, cmap=plt.cm.gray)
    if n > 1:
        plt.subplot(n, 2, 3)
        if axis_off:
            plt.axis('off')
        plt.title("Original annotation")
        plt.imshow(ann_org, cmap=plt.cm.gray)
        plt.subplot(n, 2, 4)
        if axis_off:
            plt.axis('off')
        plt.title(title_aug)
        plt.imshow(ann_aug, cmap=plt.cm.gray)
    plt.tight_layout()
    plt.show()


def show_augmentation(train):
    img_id = 12
    slice_id = 8

    imgs_org = train[img_id][0]
    anns_org = train[img_id][1]

    img_org = train[img_id][0][:, :, slice_id]
    ann_org = train[img_id][1][:, :, slice_id]

    # Rotation
    imgs_aug, anns_aug = rotate(imgs_org, anns_org, theta=10)
    plot_augmentation(img_org, ann_org, imgs_aug[:, :, slice_id], anns_aug[:, :, slice_id], title_aug='Rotation')

    # Scaling
    imgs_aug, anns_aug = scale_and_crop(imgs_org, anns_org)
    plot_augmentation(img_org, ann_org, imgs_aug[:, :, slice_id], anns_aug[:, :, slice_id], title_aug='Scaling')

    # Gray value variation
    imgs_aug, anns_aug = grayscale_variation(imgs_org, anns_org)
    plot_augmentation(img_org, [], imgs_aug[:, :, slice_id], [], title_aug='Gray variation')

    # Elastic deformation (smooth dense deformation field)
    imgs_aug, anns_aug = elastic_deformation(imgs_org, anns_org)
    plot_augmentation(img_org, ann_org, imgs_aug[:, :, slice_id, 0], anns_aug[:, :, slice_id, 0],
                      title_aug='Elastic deformation')

    # new augmentation intensity shift
    imgs_aug, anns_aug = intensity_shift(imgs_org, anns_org)
    # plot_augmentation(img_org, ann_org, imgs_aug[:, :, slice_id, 0], anns_aug[:, :, slice_id, 0],
    #                   title_aug='Intensity shift')
    plot_augmentation(img_org, [], imgs_aug[:, :, slice_id], [], title_aug='Intensity shift')
