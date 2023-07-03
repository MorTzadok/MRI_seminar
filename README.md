#                                                                  MRI Seminar Project
#                                            3D U-Net model for segmentation of prostate structures

We based our seminar project on Janko Ondras's work on [Prostate MRI Segmentation](https://github.com/jancio/3D-U-Net-Prostate-Segmentation)

Given a 3D MRI scan, the aim is to automatically annotate the peripheral zone (PZ) and central gland (CG) regions, as shown here:


![image](https://github.com/MorTzadok/MRI_seminar/assets/104845635/34e3f042-acb7-430f-9be3-2323d55498e1)


![](./figs/segmentation_task.png)

We recommend going through the PowerPoint presentation attached for further explanation.


# Regenerating the results 
The following instructions will help you run the code files to get similar results to ours.
## 1. Downloading the data

Download the original data from the [challenge website](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21267207) following the instructions in the *data access* section.

 The data should be extracted into following directories:

- NCI-ISBI: containing the DICOMS in directories TRAIN, LEADERBOARD, TEST.

- NRRD: containing nrrd files in directories Training, Leaderboard, Test.

## 2. Activating the enviroment
```
conda env create --file environment.yml
conda activate mri_env
```

## 3. Creating data files
In the file ```data.py``` enable the functions:
```
create_data() # to create the train/valid pickles
apply_histogram_equalisation_to_dataset(train, valid, test) # to apply equalization and create the heq_train/valid pickles
```
(there are more functions for visualizing the augmentations and annotations, but they are not mandatory)

## 4. Data aumentation

In the ```model.py``` file, there is a function called ```pre_process_data``` that we call in the ```main()``` function to create the augmentation for the run.
You can generate pickle files for both the original augmentations or the updated augmentations with our addition of intensity shift.

1. if you want to regenerate the  original augmentations:
   
   a. Make sure the function ```pre_process_data``` is not in comment, and the ```augment_len=10```.

   b. In ```data_augmentation.py`` make sure the variable ```p_intense``` in the function ```get_random_perturbation```` is 0.

   c. Save the augmented data in your chosen name in lines 259, 260 in the function ```pre_process_data``` in ```model.py```

   d. For future uses of the data, you can load the saved pickles with the right names in lines 610, 611 in ```model.py```

   
3. if you want to regenerate our augmentations with intensity shift:

   a. Make sure the function ```pre_process_data``` is not in comment, and the ```augment_len=10```.

   b. In ```data_augmentation.py`` make sure the variable ```p_intense``` in the function ```get_random_perturbation```` is 0.6.

   c. For future uses of the data, you can load the saved pickles with the right names in lines 610, 611 in ```model.py```

## Running train and inference

In the file ```model.py```, in the ```main()``` function there are the parameters for the run.

In order to run a new train and inference:

Set the parameters as you would like and in the parameter ```MODEL_NAME``` write your own name of the training run. this will create inside a tf directory a directory     with the model name containing the dirs:
   - net: containing the network meta and index files for each saved epoch, and checkpoint file.
   - res: containing gifs of the best and worst results and the predictions of the inference
   - hist: containing npz files with the history of every saved epoch and images with the plot of loss and iou through the run.
   - log: containing log files of the run.

After you chose the augmentations to generate and the parameters for the training, run ```model.main()```
After the run all the results will be in the directories, and the results on the inference will be printed.
