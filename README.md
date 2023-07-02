#                                                                  MRI Seminar Project
#                                            3D U-Net model for segmentation of prostate structures

We based our seminar project on Janko Ondras's work on [Prostate MRI Segmentation](https://github.com/jancio/3D-U-Net-Prostate-Segmentation)

Given a 3D MRI scan, the aim is to automatically annotate the peripheral zone (PZ) and central gland (CG) regions, as shown here:


![image](https://github.com/MorTzadok/MRI_seminar/assets/104845635/34e3f042-acb7-430f-9be3-2323d55498e1)


![](./figs/segmentation_task.png)

We recommend going through the PowerPoint presentation attached for further explanation.


# The Data
We used the [NCI-ISBI 2013 challenge Automated Segmentation of Prostate Structures](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures)

### Explanation for the data files of the project:

The pickles folder contains:
 - train.pkl, valid.pkl : train and validation before preprocessing
 - heq_train.pkl, heq_valid.pkl : train and validation after equalization preproccessing
 - train_run.pkl, valid_run.pkl : train and validation after the original data augmentation
 - train_run_aug_cng.pkl, valid_run_aug_cng.pkl : train and validation after our version of data augmentation
    
We saved the preprocessed data in the "pickles" folder for convenient future use.
If you want to run the preprocessing yourself, you can download the original data from the challenge website into the following directories:




# Regenerating the results 
The following instructions will help you run the code files to get similar results to ours.


