################################# NOTES ######################################
# new env - new_mri2 (tensoflow vwersion was downgraded)
# changed model_name to be the same
# changed line 294
##############################################################################

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
from tqdm import tqdm
from collections import Counter
from data_augmentation import *
from utils import *

from visualization import *
# from tensorflow.python.framework import ops
# Creatig the Unet class - section 5

import os

os.environ['KMP_WARNINGS'] = 'off'
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'
# Verify GPU detection
gpu_devices = tf.test.gpu_device_name()
if gpu_devices:
    print('GPU found:', gpu_devices)
else:
    print('No GPU found.')

class UNetwork():

    def conv_batch_relu(self, tensor, filters, kernel=[3, 3, 3], stride=[1, 1, 1], is_training=True):
        # Produces the conv_batch_relu combination as in the paper
        padding = 'valid'
        if self.should_pad: padding = 'same'

        conv = tf.layers.conv3d(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init)
        conv = tf.layers.batch_normalization(conv, training=is_training)
        conv = tf.nn.relu(conv)
        return conv

    def upconvolve(self, tensor, filters, kernel=2, stride=2, scale=4, activation=None):
        # Upconvolution - two different implementations: the first is as suggested in the original Unet paper and the second is a more recent version
        # Needs to be determined if these do the same thing
        padding = 'valid'
        if self.should_pad: padding = 'same'
        # upsample_routine = tf.keras.layers.UpSampling3D(size = (scale,scale,scale)) # Uses tf.resize_images
        # tensor = upsample_routine(tensor)
        # conv = tf.layers.conv3d(tensor, filters, kernel, stride, padding = 'same',
        #                                 kernel_initializer = self.base_init, kernel_regularizer = self.reg_init)
        # use_bias = False is a tensorflow bug
        conv = tf.layers.conv3d_transpose(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                          use_bias=False,
                                          kernel_initializer=self.base_init, kernel_regularizer=self.reg_init)
        return conv

    def centre_crop_and_concat(self, prev_conv, up_conv):
        # If concatenating two different sized Tensors, centre crop the first Tensor to the right size and concat
        # Needed if you don't have padding
        p_c_s = prev_conv.get_shape()
        u_c_s = up_conv.get_shape()
        offsets = np.array([0, (p_c_s[1] - u_c_s[1]) // 2, (p_c_s[2] - u_c_s[2]) // 2,
                            (p_c_s[3] - u_c_s[3]) // 2, 0], dtype=np.int32)
        size = np.array([-1, u_c_s[1], u_c_s[2], u_c_s[3], p_c_s[4]], np.int32)
        prev_conv_crop = tf.slice(prev_conv, offsets, size)
        up_concat = tf.concat((prev_conv_crop, up_conv), 4)
        return up_concat

    def __init__(self, base_filt=8, in_depth=INPUT_DEPTH, out_depth=OUTPUT_DEPTH,
                 in_size=INPUT_SIZE, out_size=OUTPUT_SIZE, num_classes=OUTPUT_CLASSES,
                 learning_rate=0.001, print_shapes=True, drop=0.2, should_pad=False, simpleUNet=False):
        # Initialise your model with the parameters defined above
        # Print-shape is a debug shape printer for convenience
        # Should_pad controls whether the model has padding or not
        # Base_filt controls the number of base conv filters the model has. Note deeper analysis paths have filters that are scaled by this value
        # Drop specifies the proportion of dropped activations

        self.base_init = tf.truncated_normal_initializer(stddev=0.1)  # Initialise weights
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)  # Initialise regularisation (was useful)

        # self.base_init = tf.compat.v1.truncated_normal_initializer(stddev=0.1) # Initialise weights
        # self.reg_init = tf.compat.v1.layers.l2_regularizer(0.1) # Initialise regularisation (was useful)

        self.should_pad = should_pad  # To pad or not to pad, that is the question
        self.drop = drop  # Set dropout rate

        with tf.compat.v1.variable_scope('3DuNet'):
            self.training = tf.compat.v1.placeholder(tf.bool)
            self.do_print = print_shapes
            self.model_input = tf.compat.v1.placeholder(tf.float32, shape=(None, in_depth, in_size, in_size, 1))
            # Define placeholders for feed_dict
            self.model_labels = tf.compat.v1.placeholder(tf.int32, shape=(None, out_depth, out_size, out_size, 1))
            labels_one_hot = tf.squeeze(tf.one_hot(self.model_labels, num_classes, axis=-1), axis=-2)

            if self.do_print:
                print('Input features shape', self.model_input.get_shape())
                print('Labels shape', labels_one_hot.get_shape())

            # Level zero
            conv_0_1 = self.conv_batch_relu(self.model_input, base_filt, is_training=self.training)
            conv_0_2 = self.conv_batch_relu(conv_0_1, base_filt * 2, is_training=self.training)
            # Level one
            max_1_1 = tf.layers.max_pooling3d(conv_0_2, [1, 2, 2], [1, 2, 2])  # Stride, Kernel previously [2,2,2]
            conv_1_1 = self.conv_batch_relu(max_1_1, base_filt * 2, is_training=self.training)
            conv_1_2 = self.conv_batch_relu(conv_1_1, base_filt * 4, is_training=self.training)
            conv_1_2 = tf.layers.dropout(conv_1_2, rate=self.drop, training=self.training)
            # Level two
            max_2_1 = tf.layers.max_pooling3d(conv_1_2, [1, 2, 2], [1, 2, 2])  # Stride, Kernel previously [2,2,2]
            conv_2_1 = self.conv_batch_relu(max_2_1, base_filt * 4, is_training=self.training)
            conv_2_2 = self.conv_batch_relu(conv_2_1, base_filt * 8, is_training=self.training)
            conv_2_2 = tf.layers.dropout(conv_2_2, rate=self.drop, training=self.training)

            if simpleUNet:
                # Level one
                up_conv_2_1 = self.upconvolve(conv_2_2, base_filt * 8, kernel=2,
                                              stride=[1, 2, 2])  # Stride previously [2,2,2]
            else:
                # Level three
                max_3_1 = tf.layers.max_pooling3d(conv_2_2, [1, 2, 2], [1, 2, 2])  # Stride, Kernel previously [2,2,2]
                conv_3_1 = self.conv_batch_relu(max_3_1, base_filt * 8, is_training=self.training)
                conv_3_2 = self.conv_batch_relu(conv_3_1, base_filt * 16, is_training=self.training)
                conv_3_2 = tf.layers.dropout(conv_3_2, rate=self.drop, training=self.training)
                # Level two
                up_conv_3_2 = self.upconvolve(conv_3_2, base_filt * 16, kernel=2,
                                              stride=[1, 2, 2])  # Stride previously [2,2,2]
                concat_2_1 = self.centre_crop_and_concat(conv_2_2, up_conv_3_2)
                conv_2_3 = self.conv_batch_relu(concat_2_1, base_filt * 8, is_training=self.training)
                conv_2_4 = self.conv_batch_relu(conv_2_3, base_filt * 8, is_training=self.training)
                conv_2_4 = tf.layers.dropout(conv_2_4, rate=self.drop, training=self.training)
                # Level one
                up_conv_2_1 = self.upconvolve(conv_2_4, base_filt * 8, kernel=2,
                                              stride=[1, 2, 2])  # Stride previously [2,2,2]

            concat_1_1 = self.centre_crop_and_concat(conv_1_2, up_conv_2_1)
            conv_1_3 = self.conv_batch_relu(concat_1_1, base_filt * 4, is_training=self.training)
            conv_1_4 = self.conv_batch_relu(conv_1_3, base_filt * 4, is_training=self.training)
            conv_1_4 = tf.layers.dropout(conv_1_4, rate=self.drop, training=self.training)
            # Level zero
            up_conv_1_0 = self.upconvolve(conv_1_4, base_filt * 4, kernel=2,
                                          stride=[1, 2, 2])  # Stride previously [2,2,2]
            concat_0_1 = self.centre_crop_and_concat(conv_0_2, up_conv_1_0)
            conv_0_3 = self.conv_batch_relu(concat_0_1, base_filt * 2, is_training=self.training)
            conv_0_4 = self.conv_batch_relu(conv_0_3, base_filt * 2, is_training=self.training)
            conv_0_4 = tf.layers.dropout(conv_0_4, rate=self.drop, training=self.training)
            conv_out = tf.layers.conv3d(conv_0_4, OUTPUT_CLASSES, [1, 1, 1], [1, 1, 1], padding='same')
            self.predictions = tf.expand_dims(tf.argmax(conv_out, axis=-1), -1)

            # Note, this can be more easily visualised in a tool like tensorboard; Follows exact same format as in Paper.

            if self.do_print:
                print('Model Convolution output shape', conv_out.get_shape())
                print('Model Argmax output shape', self.predictions.get_shape())

            do_weight = True
            loss_weights = [0.00439314, 0.68209101, 0.31351585]  # see section 1.4 # instead of [1, 150, 100, 1.0]

            # old loss
            # Weighted cross entropy: approach adapts following code: https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
            ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=conv_out, labels=labels_one_hot)
            if do_weight:
                weighted_loss = tf.reshape(tf.constant(loss_weights),
                                           [1, 1, 1, 1, num_classes])  # Format to the right size
                weighted_one_hot = tf.reduce_sum(weighted_loss * labels_one_hot, axis=-1)
                ce_loss = ce_loss * weighted_one_hot
            self.loss = tf.reduce_mean(ce_loss)  # Get loss

            self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            self.extra_update_ops = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.UPDATE_OPS)  # Ensure correct ordering for batch-norm to work
            with tf.control_dependencies(self.extra_update_ops):
                self.train_op = self.trainer.minimize(self.loss)

            # ## dice loss
            # epsilon = 1e-5  # Small constant to avoid division by zero
            #
            # dice_loss = 1.0  # Initializing the dice loss
            #
            # # Computing the intersection and sum of the predicted and ground truth masks
            # intersection = tf.reduce_sum(conv_out * labels_one_hot, axis=[1, 2, 3])
            # sum_masks = tf.reduce_sum(conv_out + labels_one_hot, axis=[1, 2, 3])
            #
            # # Computing the dice coefficient for each class
            # dice_coefficient = (2 * intersection + epsilon) / (sum_masks + epsilon)
            #
            # # Computing the dice loss for each class
            # class_loss = 1 - dice_coefficient
            #
            # # Weighting the dice loss for each class
            # if do_weight:
            #     class_loss = class_loss * tf.constant(loss_weights)
            #
            # # Computing the mean dice loss across all classes
            # dice_loss = tf.reduce_mean(class_loss)
            #
            # self.loss = dice_loss  # Set the loss as the dice loss
            #
            # self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            #
            # self.extra_update_ops = tf.compat.v1.get_collection(
            #     tf.compat.v1.GraphKeys.UPDATE_OPS)  # Ensure correct ordering for batch-norm to work
            # with tf.control_dependencies(self.extra_update_ops):
            #     self.train_op = self.trainer.minimize(self.loss)



def get_data_raw(data, i, batch_size):
    # Return separated x,y data from the i-th batch with the given batch size (batch_size)
    return [x[0] for x in data[i:i + batch_size]], [y[1] for y in data[i:i + batch_size]]


def get_pred_iou(predictions, lbl_original, ret_full=False, reswap=False):
    # Get mean_iou for full batch
    iou = []
    for i in range(len(lbl_original)):
        pred_cur = np.squeeze(predictions[i])
        metric = get_mean_iou(pred_cur, lbl_original[i], ret_full=ret_full, reswap=reswap)
        iou.append(metric)
    if ret_full:
        return np.mean(iou, axis=0)
    else:
        return np.mean(iou)

    # train the model


def pre_process_data(train, valid, augment_len=10):
    # pre process augmentation
    # Training set: scaling & augmentation
    print('--------- Preprocessing the Data ---------------')

    train_run = []
    for i in train:
        (vox, lbl) = get_scaled_input(i)
        train_run.append((vox, lbl))
        for j in range(augment_len):
            vox_a, lbl_a = get_random_perturbation(vox, lbl)
            train_run.append((vox_a, lbl_a))

    # Validation set: just scaling, no augmentation
    valid_run = []
    for i in valid:
        (vox, lbl) = get_scaled_input(i)
        valid_run.append((vox, lbl))

    pickle.dump(file=open('/home/student/Mor_MRI/pickles/train_run_aug_cng.pkl', 'wb'), obj=train_run)
    pickle.dump(file=open('/home/student/Mor_MRI/pickles/valid_run_aug_cng.pkl', 'wb'), obj=valid_run)

    return train_run, valid_run

def create_model_dir(save_path, logs_path, hist_path, res_path):
    '''
    creating the relevant dirs for the run
    :param save_path:
    :param logs_path:
    :param hist_path:
    :param res_path:
    :return:
    '''
    print('--------- creating dirs -----------------')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(hist_path):
        os.makedirs(hist_path)
    if not os.path.exists(res_path):
        os.makedirs(res_path)



def visualize_training_history(model_name,hist_path):
    '''
    showing the loss and iou plots of the run, and outputing more info
    :param model_name:
    :param hist_path:
    :return:
    '''
    print('--------- Visualizing Training -----------------')

    print('Showing history')
    hist = np.load(hist_path + model_name + '.npz')

    train_losses = hist['train_losses']
    val_losses = hist['val_losses']
    val_IOUs = hist['val_IOUs']
    train_times = hist['train_times']

    # Show training history
    mean_val_IOUs = [np.mean(iou) for iou in val_IOUs]
    print("Minimum validation loss: ", np.min(val_losses), " at epoch ", np.argmin(val_losses))
    print("Maximum validation IOU: ", np.max(mean_val_IOUs), " at epoch ", np.argmax(mean_val_IOUs))
    print("Last validation loss: ", val_losses[-1])
    print("Last validation IOU: ", mean_val_IOUs[-1])

    # Loss Curves
    plt.figure()
    x = np.arange(len(train_losses))
    plt.plot(x, train_losses, 'r-')
    plt.plot(x, val_losses, 'b-')
    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel(' Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.1)
    plt.title('Loss for ' + model_name )
    plt.savefig(hist_path + 'Loss_for_' + model_name, dpi=300)
    plt.show()

    # Validation IOU
    plt.figure()
    plt.plot(x, mean_val_IOUs, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Validatio IOU')
    plt.title('validation IOU for ' + model_name )
    plt.savefig(hist_path + 'Validation_IOU_for_' + model_name, dpi=300)
    plt.show()


def inference(test,
              unet,
              model_name,
              save_path,
              batch_size,
              res_path):
    '''
    running inference on the test set, saving the output and creating gif of the results.
    :param test:
    :param unet:
    :param model_name:
    :param save_path:
    :param batch_size:
    :param res_path:
    :return:
    '''
    print('----------------- Inference --------------------')

    # unet = UNetwork(drop=dropout, base_filt=base_filt, should_pad=True)

    config = tf.compat.v1.ConfigProto()
    test_predictions = []
    with tf.compat.v1.Session(config=config) as sess:
        # tf.initialize_all_variables().run(session=sess)

        print('Loading saved model ...')
        restorer = tf.compat.v1.train.import_meta_graph(save_path + model_name + '.meta')
        restorer.restore(sess, save_path + model_name)
        print("Model sucessfully restored")
        pred_out = []  # Predictions for each test scan
        x_orig = []  # list of non-scaled scans
        y_orig = []  # list of non-scaled annotations
        x_in = []
        y_in = []
        i = 0
        iou_out = []  # IOUs for each test scan

        while i < len(test):  # Iterate over batches
            x_batch = []
            y_batch = []
            for j in range(i, min(len(test), i + batch_size)):  # Iterate over samples within the batch
                x_orig.append(np.copy(test[j][0]))
                y_orig.append(np.copy(test[j][1]))
                x_cur, y_cur = get_scaled_input(test[j])
                x_batch.append(x_cur)
                y_batch.append(y_cur)
            if len(x_batch) == 0: break
            print('Processing ', i)
            x_in = x_in + x_batch
            y_in = y_in + y_batch
            test_dict = {
                unet.training: False,  # Whether to perform batch-norm at inference (Paper says this would be useful)
                unet.model_input: x_batch,
                unet.model_labels: y_batch
            }
            test_predictions = np.squeeze(sess.run([unet.predictions], feed_dict=test_dict))
            if len(x_batch) == 1:
                pred_out.append(test_predictions)
            else:
                pred_out.extend([np.squeeze(test_predictions[z, :, :, :]) for z in list(range(len(x_batch)))])
            i += batch_size

        for i in range(len(y_orig)):
            iou = get_mean_iou(pred_out[i], y_orig[i], ret_full=True)
            print('Test scan', i, ': IOUs: ', iou, 'Mean: ', np.mean(iou))
            iou_out.append(np.mean(iou))
        print('Mean test IOU', np.mean(iou_out), 'Std IOU', np.std(iou_out))  # mean over all test scans

    # Save test predictions
    pickle.dump(file=open('/home/student/Mor_MRI/pickles/pred_' + model_name + '.pkl', 'wb'), obj=pred_out)

    x_orig = []  # list of non-scaled scans
    y_orig = []
    for i in range(len(test)):
        x_orig.append(np.copy(test[i][0]))
        y_orig.append(np.copy(test[i][1]))

    iou_out = []  # IOUs for each test scan
    for i in range(len(y_orig)):
        iou = get_mean_iou(pred_out[i], y_orig[i], ret_full=True)
        print('Test scan', i, ': IOUs: ', iou, 'Mean: ', np.mean(iou))
        iou_out.append(np.mean(iou))
    print('\nMean test IOU', np.mean(iou_out), 'Std IOU', np.std(iou_out))  # mean over all test scans
    best_pred_image_id = np.argmax(iou_out)
    worst_pred_image_id = np.argmin(iou_out)
    print("Best prediction on the test scan: ", best_pred_image_id)
    print("Worst prediction on the test scan: ", worst_pred_image_id)

    #best predictions
    best_x_cur = x_orig[best_pred_image_id]
    best_y_cur = y_orig[best_pred_image_id]
    best_y_upscale = upscale_segmentation(swap_axes(pred_out[best_pred_image_id][:, :, :, np.newaxis]), np.shape(best_x_cur))
    # multi_slice_viewer(best_x_cur, best_y_cur, best_y_upscale)  # View  images, labels and predictions together
    make_gif(model_name, res_path, 'Best', best_x_cur, best_y_cur, best_y_upscale)
    # worst prediction
    worst_x_cur = x_orig[worst_pred_image_id]
    worst_y_cur = y_orig[worst_pred_image_id]
    worst_y_upscale = upscale_segmentation(swap_axes(pred_out[worst_pred_image_id][:, :, :, np.newaxis]), np.shape(worst_x_cur))
    # multi_slice_viewer(worst_x_cur, worst_y_cur, worst_y_upscale)  # View  images, labels and predictions together - works only in notebook
    make_gif(model_name, res_path, 'Worst', worst_x_cur, worst_y_cur, worst_y_upscale)


    pickle.dump(file=open(res_path + 'best_x_cur.pickle', 'wb'), obj=best_x_cur)
    pickle.dump(file=open(res_path + 'best_y_cur.pickle', 'wb'), obj=best_y_cur)
    pickle.dump(file=open(res_path + 'best_y_upscale.pickle', 'wb'), obj=best_y_upscale)
    pickle.dump(file=open(res_path + 'worst_x_cur.pickle', 'wb'), obj=worst_x_cur)
    pickle.dump(file=open(res_path + 'worst_y_cur.pickle', 'wb'), obj=worst_y_cur)
    pickle.dump(file=open(res_path + 'worst_y_upscale.pickle', 'wb'), obj=worst_y_upscale)

    return


def train_model(train_run,
                valid_run,
                test,
                valid,
                lr,
                dropout,
                base_filt,
                batch_size,
                patience,
                num_epochs,
                num_iter,
                model_name,
                save_path,
                logs_path,
                hist_path,
                res_path,
                save_each,
                load_model=True):
    '''
    training with the chosen parametes. inference is inside.
    :param train_run:
    :param valid_run:
    :param test:
    :param valid:
    :param lr:
    :param dropout:
    :param base_filt:
    :param batch_size:
    :param patience:
    :param num_epochs:
    :param num_iter:
    :param model_name:
    :param save_path:
    :param logs_path:
    :param hist_path:
    :param res_path:
    :param save_each:
    :param load_model:
    :return:
    '''

    print('--------- Starting to Train --------------------')


    print(num_iter, " iterations per epoch")
    tf.compat.v1.reset_default_graph()
    unet = UNetwork(drop=dropout, base_filt=base_filt, should_pad=True, learning_rate=lr)  # MODEL DEFINITION
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=4 * patience)
    config = tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(config=config) as sess:
        writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())
        if load_model:
            print('Trying to load saved model...')
            try:
                print('Loading from: ', save_path + model_name + '.meta')
                restorer = tf.compat.v1.train.import_meta_graph(save_path + model_name + '.meta')
                restorer.restore(sess, tf.train.latest_checkpoint(save_path))
                print("Model sucessfully restored")
            except IOError:
                sess.run(init)
                print("No previous model found, running default init")

        train_losses = []
        val_losses = []
        val_IOUs = []
        patience_cnt = 0
        train_times = []

        for e in range(num_epochs):

            start_time = time.time()

            # Shuffle the training data
            random.shuffle(train_run)

            curr_train_loss = []

            for i in tqdm(range(num_iter)):  # Iterate over batches within the epoch
                print('Current epoch: ', e, ', iteration: ', i, '/', num_iter, end='\n')
                x, y = get_data_raw(train_run, i, batch_size)
                train_dict = {
                    unet.training: True,
                    unet.model_input: x,
                    unet.model_labels: y
                }
                _, loss = sess.run([unet.train_op, unet.loss],
                                   feed_dict=train_dict)  # Train on batch and get train loss
                curr_train_loss.append(loss)

            # Evaluate train and valid loss
            train_loss = np.mean(curr_train_loss)
            x, y = get_data_raw(valid_run, 0, len(valid_run))  # scaled
            _, orig_y = get_data_raw(valid, 0, len(valid))  # non-scaled (for IOU evaluation)
            valid_dict = {
                unet.training: False,
                unet.model_input: x,
                unet.model_labels: y
            }
            val_loss = sess.run(unet.loss, feed_dict=valid_dict)  # Get valid loss

            # Predict on validation set and calculate IOU
            val_preds = np.squeeze(sess.run([unet.predictions], feed_dict=valid_dict))
            iou = get_pred_iou(val_preds, orig_y, ret_full=True)
            val_IOUs.append(iou)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print("Epoch ", e, "\t train_loss = ", train_loss, "\t val_loss = ", val_loss, "\t val_IOU = ",
                  np.mean(iou))

            if e > 0:
                start_time = train_times[-1]
                if (e + 1) % save_each == 0:  # save after 'each' epoch
                    print('Saving model at epoch: ', e)  # Save periodically
                    saver.save(sess, save_path + model_name, global_step=e)
                    # Save training history
                    np.savez(hist_path + model_name + '-' + str(e) + '.npz', train_losses=train_losses,
                             val_losses=val_losses,
                             val_IOUs=val_IOUs,
                             train_times=train_times)

                if val_losses[-1] > val_losses[-2]:
                    patience_cnt += 1
                else:
                    patience_cnt = 0

            e_time = time.time() - start_time
            print('Epoch ' + str(e) + ' running time: ' + str(e_time / 60))
            train_times.append(e_time)

            if e + 1 == num_epochs:
                saver.save(sess, save_path + model_name + '-final', global_step=e)

            if patience_cnt >= patience:
                print("Early stopping ...")
                saver.save(sess, save_path + model_name + '-final', global_step=e)
                break

    # Save training history
    final_model_name = model_name + '-final-' + str(e)

    np.savez(hist_path + final_model_name + '.npz', train_losses=train_losses, val_losses=val_losses,
             val_IOUs=val_IOUs,
             train_times=train_times)


    print('--------- Finished Training --------------------')
    visualize_training_history(final_model_name, hist_path)
    inference(test, unet, final_model_name, save_path, batch_size, res_path)





def main():
    # Load the histogram equalised dataset if needed
    print('--------- Loading the Data ---------------------')
    train = pickle.load(file=open('/home/student/Mor_MRI/pickles/heq_train.pkl', 'rb'))
    valid = pickle.load(file=open('/home/student/Mor_MRI/pickles/heq_valid.pkl', 'rb'))
    test = pickle.load(file=open('/home/student/Mor_MRI/pickles/heq_test.pkl', 'rb'))


    # train_run, valid_run = pre_process_data(train, valid, augment_len=10)

    train_run = pickle.load(file=open('/home/student/Mor_MRI/pickles/train_run_1.pkl', 'rb'))
    valid_run = pickle.load(file=open('/home/student/Mor_MRI/pickles/valid_run_1.pkl', 'rb'))



    #Hyper parameters
    LEARNING_RATE = 0.0005  # Model learning rate
    DROPOUT = 0.1
    BASE_FILT = 8  # Number of base filters
    BATCH_SIZE = 10  # Batch size - VRAM limited; originally 3
    PATIENCE = 5  # For early stopping: watching for validation loss increase
    NUM_EPOCHS = 30  # Maximum number of training epochs
    NUM_ITER = len(train_run) // BATCH_SIZE  # Number of training steps per epoch
    MODEL_NAME = '30e-intensity_aug_lr_0005'  # Model name to LOAD FROM (looks IN SAVE_PATH directory)
    NET_PATH = "/home/student/Mor_MRI/tf/" + MODEL_NAME + '/net/'
    LOGS_PATH = "/home/student/Mor_MRI/tf/" + MODEL_NAME + '/log/'
    HIST_PATH = "/home/student/Mor_MRI/tf/" + MODEL_NAME + '/hist/'
    RES_PATH = "/home/student/Mor_MRI/tf/" + MODEL_NAME + '/res/'

    create_model_dir(NET_PATH, LOGS_PATH, HIST_PATH, RES_PATH)


    train_model(train_run, valid_run, test, valid, LEARNING_RATE, DROPOUT, BASE_FILT, BATCH_SIZE, PATIENCE, NUM_EPOCHS,
                NUM_ITER, MODEL_NAME, NET_PATH, LOGS_PATH, HIST_PATH, RES_PATH, save_each=5)




if __name__ == '__main__':
    main()