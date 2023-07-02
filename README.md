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
 - train_run_1.pkl, valid_run_1.pkl : train and validation after our version of data augmentation
 - train_run_aug_cng.pkl, valid_run_aug_cng.pkl : train and validation after a version of data augmentation we didnt use because it had worse results
    
We saved the preprocessed data in the "pickles" folder for convenient future use.
If you want to run the preprocessing yourself, you can download the original data from the challenge website into the following directories:
- NCI-ISBI: containing the DICOMS in directories TRAIN, LEADERBOARD, TEST.
- NNRD: containing nnrd files in directories Training, Leaderboard, Test.

then you should create the enviroment using the enviroment.yml file

```
conda env create --file environment.yml
```

and then activate the enviroment

```
conda activate new_mri2
```
in the file ```data.py```` enable the functions:
```
create_data() # to create the train/valid pickles
apply_histogram_equalisation_to_dataset(train, valid, test) # to apply equalization and create the heq_train/valid pickles
```
(there are more functions for visualizing the augmentations and annotations, but they are not mandatory)


# Regenerating the results 
The following instructions will help you run the code files to get similar results to ours.
if you didn't regenerate the pickled data, create and activate the env:

```
conda env create --file environment.yml
conda activate new_mri2
```
in the file ```model.py```, in the ```main()``` function there are the parameters for the run.
in order to run a new train:

1. if you want to regenerate the augmentations:

   a. in the ```main()``` function, make sure the function ```pre_process_data``` is not in comment, and the ```augment_len=10```.

   b. in the file ```data_augmentation.py``` make sure that functions ```rotate,  grayscale_variation ``` have the original hyperparameters.

   c. make sure the ```p_intense``` in the function ```get_random_perturbation``` in ```data_augmentation``` is 0.
   
2. if you want to regenerate our augmentations with intensity shift:

   a. in the ```main()``` function, make sure the function ```pre_process_data``` is not in comment, and the ```augment_len=10```.

   b. in the file ```data_augmentation.py``` make sure that functions ```rotate,  grayscale_variation ``` have the original hyperparameters.

   c. make sure the ```p_intense``` in the function ```get_random_perturbation``` in ```data_augmentation.py``` is 0.6.

3. if you want to use the augmented train and validation set:
   in ```model.py``` in ```main()``` the lines:
   ```
   train_run = pickle.load(file=open('/home/student/Mor_MRI/pickles/train_run_aug_cng.pkl', 'rb'))
   valid_run = pickle.load(file=open('/home/student/Mor_MRI/pickles/valid_run_aug_cng.pkl', 'rb'))
   ```
   need to refer to the right paths:

   train/valid_run.pkl for the original augmentations

   train/valid_run_1.pkl for the augmentations with intensity shift

   train/valid_run_aug_cng.pkl for more changes in augmentations (worse results)


    the function ```pre_process_data``` shoul be in comment
   
5. set the parameters as you would like and in the parameter ```MODEL_NAME``` write your own name of the training run. this will create inside the tf directory a directory     with the model name containing the dirs:
   - net: containing the network meta and index files for each saved epoch, and checkpoint file.
   - res: containing gifs of the best and worst results and the predictions of the inference
   - hist: containing npz files with the history of every saved epoch and images with the plot of loss and iou through the run.
   - log: containing log files of the run.

 6. run ```model.main()```

   after the run all the results will be in the directories, and the results on the inference will be printed.
