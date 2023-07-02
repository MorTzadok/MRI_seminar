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


# if sys.version_info[0] != 3:
#     raise Exception("Python version 3 has to be used!")
#
# print("Currently using")
# print("\t numpy ", np.__version__)
# print("\t scipy ", scipy.__version__)
# print("\t matplotlib ", matplotlib.__version__)
# print("\t tensorflow ", tf.__version__)
# print("\t pydicom ", pydicom.__version__)
# print("\t nrrd ", nrrd.__version__)
# print("\t skimage ", skimage.__version__)

np.random.seed(37) # for reproducibility


def return_dcm(file_path, check_term='Prostate'):
    # Read all DCM (slices) files within a directory and order the files based on filename
    out_dcm = {}
    for dirName, subdirList, fileList in os.walk(file_path):
        c_dcm = []
        cur_name = ""
        dir_split = dirName.split("/")
        for f_chk in dir_split:
            if check_term in f_chk:
                if f_chk == '07-15-2008-NA-MRI Prostate With and Without Contrast-57538':
                    continue
                cur_name = f_chk
        for filename in fileList:
            if ".dcm" in filename.lower():
                temp = os.path.splitext(filename)[0]
                temp = temp.replace('-', '')
                # name = int(os.path.splitext(filename)[0])
                name = int(temp)
                c_dcm.append((os.path.join(dirName, filename), name))
        if len(c_dcm) > 0:
            c_dcm = sorted(c_dcm, key=lambda t: t[1])  # Sort into correct order
            if cur_name == '07-15-2008-NA-MRI Prostate With and Without Contrast-57538':
                print('hi')
            out_dcm[cur_name] = [c[0] for c in c_dcm]  # Store in dictionary
    return out_dcm


def return_nrrd(file_path):
    # Read all NRRD (annotation) files within a directory
    out_nrrd = {}
    for dirName, subdirList, fileList in os.walk(file_path):
        for filename in fileList:
            if ".nrrd" in filename.lower():
                name = filename.split('_')[0]
                name = name.split('.')[0]  # Get annotation name and store in dictionary
                out_nrrd[name] = os.path.join(dirName, filename)
    return out_nrrd


def get_dataset(data_dir, anns_dir):
    # Match DCM volumes with corresponding annotation files
    data_out = []
    shapes = {}
    d_dcm = return_dcm(data_dir)
    d_nrrd = return_nrrd(anns_dir)


    for i in d_nrrd:

        seg, opts = nrrd.read(d_nrrd[i])
        voxels = np.zeros(np.shape(seg))
        for j in range(len(d_dcm[i])):
            dicom_ref = pydicom.read_file(d_dcm[i][j])
            found = False
            chk_val = dicom_ref[("0020", "0013")].value
            # Make sure you get the right slice! This is a bizarre specification thing related to DCM dataset
            # Note, if you just use the default filename ordering you get mismatched slices!
            if int(chk_val.__str__()) - 1 < np.shape(voxels)[-1]:
                voxels[:, :, int(chk_val.__str__()) - 1] = dicom_ref.pixel_array
            else:
                print('Index: ', str(int(chk_val.__str__()) - 1), ' too large for ', i, ' skipping!')
        # Rotate and flip annotations to match volumes
        seg = scipy.ndimage.interpolation.rotate(seg, 90, reshape=False)
        for i in range(np.shape(seg)[2]):
            cur_img = np.squeeze(seg[:, :, i])
            seg[:, :, i] = np.flipud(cur_img)
        # Store volume shapes (for debug)
        if voxels.shape in shapes:
            shapes[voxels.shape] += 1
        else:
            shapes[voxels.shape] = 1
        # Saves data
        data_out.append((voxels, seg))
    return data_out


def plot_slice(slice_in, is_anns=False, num_anns=4):
    # Plot a slice of data - can either be raw image data or corresponding annotation
    slice_in = np.squeeze(slice_in)
    plt.figure()
    plt.set_cmap(plt.bone())
    if is_anns:
        plt.pcolormesh(slice_in, vmin=0, vmax=num_anns - 1)
    else:
        plt.pcolormesh(slice_in)
    plt.show()



def multi_slice_viewer(feats, anns = None, preds = None, num_classes = 4, no_axis=False):
    # Plot feats, anns, predictions in multi-slice-view
    # feats OR feats + anns OR feats + anns + preds
    # only works in notebook
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




def process_key(event):
    # Process key_press events
    fig = event.canvas.figure
    if event.key == 'j':
        for ax in fig.axes:
            previous_slice(ax)
    elif event.key == 'k':
        for ax in fig.axes:
            next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    # Go to the previous slice
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[-1]  # wrap around using %
    ax.images[0].set_array(volume[:, :, ax.index])


def next_slice(ax):
    # Go to the next slice
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[-1]
    ax.images[0].set_array(volume[:, :, ax.index])

def create_data():
    # .dcm data files
    data_train_dir = '/home/student/Mor_MRI/NCI-ISBI/TRAIN/'
    data_leader_dir = '/home/student/Mor_MRI/NCI-ISBI/LEADERBOARD/'
    data_test_dir = '/home/student/Mor_MRI/NCI-ISBI/TEST/'

    # .nrrd annotation files
    anns_train_dir = '/home/student/Mor_MRI/NNRD/Training/'
    anns_leader_dir = '/home/student/Mor_MRI/NNRD/Leaderboard/'
    anns_test_dir = '/home/student/Mor_MRI/NNRD/Test/'

    train = get_dataset(data_train_dir, anns_train_dir)
    valid = get_dataset(data_leader_dir, anns_leader_dir)
    test = get_dataset(data_test_dir, anns_test_dir)

    if not os.path.exists('./pickles'):
        os.makedirs('./pickles')
    pickle.dump(file=open('./pickles/train.pkl', 'wb'), obj=train)
    pickle.dump(file=open('./pickles/valid.pkl', 'wb'), obj=valid)
    pickle.dump(file=open('./pickles/test.pkl', 'wb'), obj=test)

    print("\nTraining scans:", len(train), "\t\t Scan slices:", np.sum([x.shape[2] for x, _ in train]),
          "\nValidation scans:", len(valid), "\t\t Scan slices:", np.sum([x.shape[2] for x, _ in valid]),
          "\nTesting scans: ", len(test), "\t\t Scan slices:", np.sum([x.shape[2] for x, _ in test]))
    print("Sample 3D scans' shapes:", train[2][0].shape, valid[1][0].shape,
          test[9][0].shape)  # as we can see these shapes vary

def do_cross_val(train, valid, test): # Whether to do cross-validation
    """
    Perform k-fold cross validation, as suggested in the 3D U-Net paper. For each fold, this needs to be run again.
    Note: **This is left as an extension and it was not performed as it would be very time-consuming. **
    TODO: continue the func for implementation
    :param train:
    :param valid:
    :param test:
    :return:
    """
    data_total = train + valid + test
    K_FOLD = 3
    VALID_FRAC = 0.25 # fraction of the training set used as validation set
    CURRENT_FOLD = 0  # need to be set to: 0, 1, ... K_FOLD-1

    val_split = len(data_total)/K_FOLD
    val_idx = CURRENT_FOLD*val_split
    train = data_total[:val_idx] + data_total[val_idx+val_split:]
    valid = train[:int(len(train)*VALID_FRAC)]
    train = train[int(len(train)*VALID_FRAC):]
    test = data_total[val_idx:val_idx+val_split]
    data_total = []

def show_class_frequency(train):
    class_freq = {0: 0, 1: 0, 2: 0}
    for i in range(len(train)):
        for j in range(train[i][1].shape[2]):
            d = Counter(train[i][1][:, :, j].flatten())
            class_freq[0] += d[0]
            class_freq[1] += d[1]
            class_freq[2] += d[2]
    print("Class frequencies in training set: ", class_freq)

    inv_class_freq = 1. / np.array([class_freq[0], class_freq[1], class_freq[2]], dtype=np.float64)
    class_weights = inv_class_freq / sum(inv_class_freq)
    print("Class weights (inversely proportional to class frequencies): ", class_weights)



def apply_histogram_equalisation_to_dataset(train, valid, test):

    hist_equalise(train)  # in-place
    hist_equalise(valid)
    hist_equalise(test)

    # Save histogram equalised dataset
    pickle.dump(file=open('./pickles/heq_train.pkl', 'wb'), obj=train)
    pickle.dump(file=open('./pickles/heq_valid.pkl', 'wb'), obj=valid)
    pickle.dump(file=open('./pickles/heq_test.pkl', 'wb'), obj=test)


def visualise_scaled_scans_and_annotations(dataset):
    img_id = 15  # Image ID to view
    x, y = get_scaled_input(dataset[img_id])  # Shows that this works - can check x,y shapes if needed
    x_swap = swap_axes(x)
    y_upscale = upscale_segmentation(swap_axes(y), np.shape(swap_axes(x)))
    multi_slice_viewer(x_swap, y_upscale)  # View scaled images and labels together

    # Compute mean iou with itself & upsampled data
    x, y = get_scaled_input(dataset[img_id])
    print('Mean IOU with itself')
    print(get_mean_iou(y, y, ret_full=True, reswap=True))
    print('Mean IOU with original labels')
    print(get_mean_iou(y, dataset[img_id][1], ret_full=True, reswap=False))


def main():

    # run only the first time
    # create_data()


    with open('./pickles/train.pkl', 'rb') as file:
        train = pickle.load(file)

    with open('./pickles/valid.pkl', 'rb') as file:
        valid = pickle.load(file)

    with open('./pickles/test.pkl', 'rb') as file:
        test = pickle.load(file)

    # run only the first time
    # apply_histogram_equalisation_to_dataset(train, valid, test)

    # show_class_frequency(train)

    show_augmentation(train)

    # visualise_scaled_scans_and_annotations(train)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()