from skimage import exposure
from skimage.util import img_as_float
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import os
import sys
import pydicom
import nrrd
import scipy.ndimage
import random
import pickle

def hist_equalise(dataset):
    # Perform histogram equalisation on each scan slice
    for i in range(len(dataset)):  # Over subjects
        for j in range(dataset[i][0].shape[2]):  # Over scan slices
            dataset[i][0][:, :, j] = exposure.equalize_hist(dataset[i][0][:, :, j])


def plot_img_and_hist(image, axes, bins=100):
    # Plot an image along with its histogram and cumulative histogram.
    # Code adapted from: http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def plot_histogram_equalisation(img_org, img_heq):
    # Plot original low contrast image and histogram eualised image, with their histograms and cumulative histograms.
    # Code adapted from: http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

    fig = plt.figure(figsize=(7, 7))
    axes = np.zeros((2, 2), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 2, 1)
    for i in range(1, 2):
        axes[0, i] = fig.add_subplot(2, 2, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 2):
        axes[1, i] = fig.add_subplot(2, 2, 3 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_org, axes[:, 0])
    ax_img.set_title('Original low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_heq, axes[:, 1])
    ax_img.set_title('Histogram equalised image')

    ax_cdf.set_ylabel('Fraction of total intensity', color='red')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    fig.tight_layout(h_pad=-1.5)
    plt.show()



INPUT_SIZE = 120  # Input feature width/height
OUTPUT_SIZE = 120  # Output feature width/height (as defined by model)
INPUT_DEPTH = 12  # Input depth
OFF_IMAGE_FILL = 0  # What to fill an image with if padding is required to make Tensor
OFF_LABEL_FILL = 0  # What to fill a label with if padding is required to make Tensor
OUTPUT_CLASSES = 3  # Number of output classes in dataset: the fourth class can be used for unlabelled datapoints (not needed for this dataset)

# Get 'natural' OUTPUT_DEPTH according to scipy method
io_zoom = OUTPUT_SIZE / INPUT_SIZE
zero_chk = np.zeros((INPUT_SIZE, INPUT_SIZE, INPUT_DEPTH))
OUTPUT_DEPTH = np.shape(scipy.ndimage.interpolation.zoom(zero_chk, io_zoom, order=1))[-1]
# Alternatively, this can be forced to match expected tensorflow output (note, extra padding is applied if depth mismatch)
OUTPUT_DEPTH = 12


def get_scaled_input(data, min_i=INPUT_SIZE, min_o=OUTPUT_SIZE, depth=INPUT_DEPTH,
                     depth_out=OUTPUT_DEPTH, image_fill=OFF_IMAGE_FILL,
                     label_fill=OFF_LABEL_FILL, n_classes=OUTPUT_CLASSES, norm_max=500):
    # Takes raw data (x, y) and scales to match desired input and output sizes to feed into Tensorflow
    # Pads and normalises input and also moves axes around to orientation expected by tensorflow

    input_scale_factor = min_i / data[0].shape[0]
    output_scale_factor = min_o / data[0].shape[0]

    vox_zoom = None
    lbl_zoom = None

    if not input_scale_factor == 1:
        vox_zoom = scipy.ndimage.interpolation.zoom(data[0], input_scale_factor, order=1)
        # Order 1 is bilinear - fast and good enough
    else:
        vox_zoom = data[0]

    if not output_scale_factor == 1:
        lbl_zoom = scipy.ndimage.interpolation.zoom(data[1], output_scale_factor, order=0)
        # Order 0 is nearest neighbours: VERY IMPORTANT as it ensures labels are scaled properly (and stay discrete)
    else:
        lbl_zoom = data[1]

    lbl_pad = label_fill * np.ones((min_o, min_o, depth_out - lbl_zoom.shape[-1]))
    lbl_zoom = np.concatenate((lbl_zoom, lbl_pad), 2)
    lbl_zoom = lbl_zoom[np.newaxis, :, :, :]

    vox_pad = image_fill * np.ones((min_i, min_i, depth - vox_zoom.shape[-1]))
    vox_zoom = np.concatenate((vox_zoom, vox_pad), 2)

    max_val = np.max(vox_zoom)
    if not np.max(vox_zoom) == 0:
        vox_zoom = vox_zoom * norm_max / np.max(vox_zoom)

    vox_zoom = vox_zoom[np.newaxis, :, :, :]

    vox_zoom = np.swapaxes(vox_zoom, 0, -1)
    lbl_zoom = np.swapaxes(lbl_zoom, 0, -1)
    # Swap axes

    return vox_zoom, lbl_zoom


def upscale_segmentation(lbl, shape_desired):
    # Returns scaled up label for a given input label and desired shape. Required for Mean IOU calculation

    scale_factor = shape_desired[0] / lbl.shape[0]
    lbl_upscale = scipy.ndimage.interpolation.zoom(lbl, scale_factor, order=0)
    # Order 0 EVEN more important here
    lbl_upscale = lbl_upscale[:, :, :shape_desired[-1]]
    if lbl_upscale.shape[-1] < shape_desired[-1]:
        pad_zero = OFF_LABEL_FILL * np.zeros(
            (shape_desired[0], shape_desired[1], shape_desired[2] - lbl_upscale.shape[-1]))
        lbl_upscale = np.concatenate((lbl_upscale, pad_zero), axis=-1)
    return lbl_upscale


def get_label_accuracy(pred, lbl_original):
    # Get pixel-wise labelling accuracy (DEMO metric)

    # Swap axes back
    pred = swap_axes(pred)
    pred_upscale = upscale_segmentation(pred, np.shape(lbl_original))
    return 100 * np.sum(np.equal(pred_upscale, lbl_original)) / np.prod(lbl_original.shape)


def get_mean_iou(pred, lbl_original, num_classes=OUTPUT_CLASSES, ret_full=False, reswap=False):
    # Get mean IOU between input predictions and target labels. Note, method implicitly resizes as needed
    # Ret_full - returns the full iou across all classes
    # Reswap - if lbl_original is in tensorflow format, swap it back into the format expected by plotting tools (+ format of raw data)

    # Swap axes back
    pred = swap_axes(pred)
    if reswap:
        lbl_original = swap_axes(lbl_original)
    pred_upscale = upscale_segmentation(pred, np.shape(lbl_original))
    iou = [1] * num_classes
    for i in range(num_classes):
        test_shape = np.zeros(np.shape(lbl_original))
        test_shape[pred_upscale == i] = 1
        test_shape[lbl_original == i] = 1
        full_sum = int(np.sum(test_shape))
        test_shape = -1 * np.ones(np.shape(lbl_original))
        test_shape[lbl_original == i] = pred_upscale[lbl_original == i]
        t_p = int(np.sum(test_shape == i))
        if not full_sum == 0:
            iou[i] = t_p / full_sum
    if ret_full:
        return iou
    else:
        return np.mean(iou)


def swap_axes(pred):
    # Swap those axes
    pred = np.swapaxes(pred, -1, 0)
    pred = np.squeeze(pred)
    return pred