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
    """
    return all the DICOM files in the parent dir
    :param file_path: the parent dir with DICOMS
    :param check_term: term that all the relevant files have
    :return: all the relevant dicom files
    """
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
    '''
    Read all NRRD (annotation) files within a directory
    :param file_path: nrrd parent dir
    :return: relevant nrrd files
    '''
    out_nrrd = {}
    for dirName, subdirList, fileList in os.walk(file_path):
        for filename in fileList:
            if ".nrrd" in filename.lower():
                name = filename.split('_')[0]
                name = name.split('.')[0]  # Get annotation name and store in dictionary
                out_nrrd[name] = os.path.join(dirName, filename)
    return out_nrrd


def get_dataset(data_dir, anns_dir):
    '''
    Match DCM volumes with corresponding annotation files
    :param data_dir: parent dicom dir
    :param anns_dir: parent nrrd dir
    :return:
    '''
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


def create_data():
    '''
    main function that creates the data out of the dirs downloaded.
    make sure your dirs are the same, or change the path here.
    saves pickle files with the extracted data.
    :return:
    '''
    # .dcm data files
    data_train_dir = '/home/student/Mor_MRI/NCI-ISBI/TRAIN/'
    data_leader_dir = '/home/student/Mor_MRI/NCI-ISBI/LEADERBOARD/'
    data_test_dir = '/home/student/Mor_MRI/NCI-ISBI/TEST/'

    # .nrrd annotation files
    anns_train_dir = '/home/student/Mor_MRI/NRRD/Training/'
    anns_leader_dir = '/home/student/Mor_MRI/NRRD/Leaderboard/'
    anns_test_dir = '/home/student/Mor_MRI/NRRD/Test/'

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


def show_class_frequency(train):
    '''
    showing the frequencies of each class for the weighted loss
    :param train:
    :return:
    '''
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
    '''
    applying equalization and saving it in pkl file
    :param train:
    :param valid:
    :param test:
    :return:
    '''

    hist_equalise(train)  # in-place
    hist_equalise(valid)
    hist_equalise(test)

    # Save histogram equalised dataset
    pickle.dump(file=open('./pickles/heq_train.pkl', 'wb'), obj=train)
    pickle.dump(file=open('./pickles/heq_valid.pkl', 'wb'), obj=valid)
    pickle.dump(file=open('./pickles/heq_test.pkl', 'wb'), obj=test)



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







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
