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
from data_augmentation import *
from utils import *
import imageio
from IPython.display import Image, display


def plot_slice(slice_in, is_anns=False, num_anns=4):
    '''
    Plot a slice of data - can either be raw image data or corresponding annotation
    :param slice_in:
    :param is_anns:
    :param num_anns:
    :return:
    '''

    slice_in = np.squeeze(slice_in)
    plt.figure()
    plt.set_cmap(plt.bone())
    if is_anns:
        plt.pcolormesh(slice_in, vmin=0, vmax=num_anns - 1)
    else:
        plt.pcolormesh(slice_in)
    plt.show()


def process_key(event):
    '''
    Process key_press events
    :param event:
    :return:
    '''

    fig = event.canvas.figure
    if event.key == 'j':
        for ax in fig.axes:
            previous_slice(ax)
    elif event.key == 'k':
        for ax in fig.axes:
            next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    '''
    Go to the previous slice
    :param ax:
    :return:
    '''

    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[-1]  # wrap around using %
    ax.images[0].set_array(volume[:, :, ax.index])


def next_slice(ax):
    '''
    Go to the next slice
    :param ax:
    :return:
    '''

    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[-1]
    ax.images[0].set_array(volume[:, :, ax.index])


def multi_slice_viewer(feats, anns = None, preds = None, num_classes = 4, no_axis=False):
    '''
    # Plot feats, anns, predictions in multi-slice-view
    # feats OR feats + anns OR feats + anns + preds
    # only works in notebook
    # Multi-slice view code extracted and adapted from:
    # https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
    '''
    if anns is None:
        fig, ax = plt.subplots()
        ax.volume = feats
        ax.index = feats.shape[-1] // 2
        ax.imshow(feats[:, :, ax.index],  cmap='bone')
        fig.canvas.mpl_connect('key_press_event', process_key)
    else:
        if preds is None:
            fig, axarr = plt.subplots(1, 2)
            plt.tight_layout()
            axarr[0].volume = feats
            axarr[0].index = 0
            axarr[0].imshow(feats[:, :, axarr[0].index],  cmap='bone')
            axarr[0].set_title('Scans')
            axarr[1].volume = anns
            axarr[1].index = 0
            axarr[1].imshow(anns[:, :, axarr[1].index],  cmap='bone', vmin = 0, vmax = num_classes)
            axarr[1].set_title('Annotations')
            fig.canvas.mpl_connect('key_press_event', process_key)
        else:
            fig, axarr = plt.subplots(1, 3)
            plt.tight_layout()
            axarr[0].volume = feats
            axarr[0].index = 0
            axarr[0].imshow(feats[:, :, axarr[0].index],  cmap='bone')
            axarr[0].set_title('Scans')
            axarr[1].volume = anns
            axarr[1].index = 0
            axarr[1].imshow(anns[:, :, axarr[1].index],  cmap='bone', vmin = 0, vmax = num_classes)
            axarr[1].set_title('Annotations')
            axarr[2].volume = preds
            axarr[2].index = 0
            axarr[2].imshow(preds[:, :, axarr[2].index],  cmap='bone', vmin = 0, vmax = num_classes)
            axarr[2].set_title('Predictions')
            fig.canvas.mpl_connect('key_press_event', process_key)
        if no_axis:
            for a in axarr:
                a.set_axis_off()




def make_gif(model_name, res_path, type, feats, anns, preds, num_classes=4):
    '''
    making a gif out of the sample slices
    :param model_name: for titlr
    :param res_path: file where th egif is saved
    :param type:
    :param feats: scans
    :param anns: labels
    :param preds: predictions
    :param num_classes: for the segmentation
    :return:
    '''
    save_path = res_path + 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Assuming your ndarrays are named 'images1', 'images2', and 'images3' with shape (384, 384, 21)
    num_images = feats.shape[2]

    # Create a list to store the subplot images
    subplot_images = []

    # Iterate through the images
    for i in range(num_images):
        # Extract the corresponding images from each ndarray
        scan = feats[:, :, i]
        ann = anns[:, :, i]
        pred = preds[:, :, i]

        # Create a subplot with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3)
        plt.tight_layout()

        # Plot the three images in the subplot
        axes[0].imshow(scan, cmap='bone')
        axes[0].set_title('scan',)

        axes[1].imshow(ann, cmap='bone', vmin=0, vmax=num_classes)
        axes[1].set_title('annotation')

        axes[2].imshow(pred, cmap='bone', vmin=0, vmax=num_classes)
        axes[2].set_title('prediction')

        subplot_title = type + ' prediction for model ' + model_name
        fig.suptitle(subplot_title, fontsize=14, fontweight='bold')

        # Remove the axis labels
        for ax in axes:
            ax.axis('off')

        # Save the subplot as an image

        if i == 0:
            plt.show()

        filename = f'{save_path}{type}_{model_name}_{i}.png'
        plt.savefig(filename, dpi=300)
        plt.close()

        # Append the image to the list of subplot images
        subplot_images.append(imageio.imread(filename))

    # Save the list of subplot images as a GIF
    imageio.mimsave(f'{res_path}{type}_{model_name}.gif', subplot_images, duration=0.2)
    display(Image(filename=f'{res_path}{type}_{model_name}.gif'))




# MODEL_NAME = '4e_try6'  # Model name to LOAD FROM (looks IN SAVE_PATH directory)
# RES_PATH = "/home/student/Mor_MRI/tf/" + MODEL_NAME + '/res/'
# valid_run = pickle.load(file=open('/home/student/Mor_MRI/pickles/valid_run.pkl', 'rb'))
# best_x_cur = pickle.load(file=open(RES_PATH + 'best_x_cur.pickle', 'rb'))
# best_y_cur = pickle.load(file=open(RES_PATH + 'best_y_cur.pickle', 'rb'))
# best_y_upscale = pickle.load(file=open(RES_PATH + 'best_y_upscale.pickle', 'rb'))
# worst_x_cur = pickle.load(file=open(RES_PATH + 'worst_x_cur.pickle', 'rb'))
# worst_y_cur = pickle.load(file=open(RES_PATH + 'worst_y_cur.pickle', 'rb'))
# worst_y_upscale = pickle.load(file=open(RES_PATH + 'worst_y_upscale.pickle', 'rb'))


# make_gif(MODEL_NAME, RES_PATH, 'Best', best_x_cur, anns=best_y_cur, preds=best_y_upscale, num_classes=4)