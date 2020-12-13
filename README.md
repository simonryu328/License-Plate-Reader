# cnn_training

## Folder Structure
This is repository is supposed to be a workspace and not a place to save the training and validation data on a permanent basis. Once images are manipulated/generated the data is saved at 
[enph353 project/CNN_data](https://drive.google.com/drive/u/0/folders/1GhAsbMGbIHb7_XyJUaurLmqfQbezjeYd)

## save rqt image
save_rqt_image.py is a script for saving the latest image published to "/R1/pi_camera/image_raw" and labelling it. This will be useful for tagging images for training the CNN. 

The syntax I have been using is:
`#Foldername/#PlateID_#Licensenumber`

The script checks how many file names match this format, increments the unique identifier and saves an image to:
`#Foldername/#PlateID_#Licensenumber_#uniqueintedifier`

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ python save_rqt_image.py simulation_data/P1_EQ07
```

## perspective transform

### perspective_transform_script.py
 a python script for taking in an off angle images of license plates (from the simulation) and flattening them. An example is as follows

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ python perspective_transform.py test/P1_EQ07_1.png
```
Will apply a perspective transform to `test/P1_EQ07`, and save a corresponding flattened image as `test/flattened/P1_EQ07_1.png`
### perspective_transform_all.sh
 a bash script for running `perspective_transform_script.py` on all files in a certain directory.

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ ./perspective_transform_all.sh test/
```
Will apply a perspective transform to every file in test, and save a corresponding flattened image in `test/flattened`

## Character Segmentation

### segment_license_plate_images.py
 a python script for taking in flattened images of license plates and seperating each character into its own file. The image is also greyscaled and the license plate ID characters are scaled down to have the same size as the license plate characters.

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ python segment_license_plate_images.py test/flattened/P1_EQ07_1.png
```
In this example, the output will be 5 files tagged:  1_#, saved to `test/flattened/segmented_parkingID`; and E_#, Q_#, 0_#, 7_# saved to `test/flattened/segemented_plate`.
### segment_all.sh
 a bash script for running `perspective_transform_script.py` on all files in a certain directory.

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ ./perspective_transform_all.sh test/flattened
```
Will apply segment image in `test/flattened`, and save a corresponding segmented characters in `test/flattened/segmented_parkingID` and `test/flattened/segmented_plate`

## plate_generator.py
This is based off the plate generator script Miti provided at `2020T1_competition/enph353/enph353_gazebo/scripts/plate_generator.py`.

This modified script generates random plate configurations and then segments these plate images into individual characters. The output is a set of segmented characters "syntethically" generated. There is no data augmentation applied.

The script takes one integer as an input argument, which specifies the amount of plates to randomly generate. The amount of output images will be 5 times the input integer ( as there are 5 characters for each plate, ignoring P which is the same for all plates).

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ python plate_generator.py 2
```
This example will produce 10 character images, 2 of which will be saved in the `./generated_data/segmented_parkingID` folder and 8 of which will be saved in the `./generated_data/segmented_plate` folder.

## CNN training
### Training Data and Validation Data
To run the the `plate_detector_cnn_training,py` script, you will need to first populate the `training_scripts/training_data` and `training_scripts/validation` data with the appropriate datasets.

The datasets that yield the best accuracy are saved here: 
[enph353 project/CNN_data/current best data.zip](https://drive.google.com/file/d/1uggD_tpvas8vq9OJPlnWmAMk11Id_mSV/view?usp=sharing). Download and extract this zip file. "Copy/paste" the data in `current best data\current best training data"` to `training_scripts/training_data` and `current best data\current best validation data"` to `training_scripts/validation_data`.

### plate_detector_cnn_training.py
To run the CNN training, run the following:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ cd training_scripts
fizzer@enph353:~/ros_ws/src/cnn_training/training_scripts$ python plate_detector_cnn_training.py
```

A few Pyplot windows will pop up. You will need to x-out these for the script to proceed.

The first window to pop shows a few examples of augmented data that will be used for training

The second and third window that pop up show loss and accuracy metrics.

The 4th window that pops up is an example of the prediction probabilities for a random character in the validation dataset.

The final window that pops shows the confusion matrix and the modified accuracy calculation. (Modified accuracy calculation - If we have an alphabetic character as an input, we can make our prediction based only off of alphabetic character probabilities. This is because we will always know if an input is an alphabetic character, based off of its location. The converse is true for numeric characters).