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
In this example, the output will be 6 files tagged P_#, 1_#, E_#, Q_#, 0_#, 7_#, with the corresponding segmented character from the plate.
### segment_all.sh
 a bash script for running `perspective_transform_script.py` on all files in a certain directory.

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ ./perspective_transform_all.sh test/flattened
```
Will apply segment image in `test/flattened`, and save a corresponding segmented characters in `test/flattened/segmented`

## plate_generator.py
This is based of the plate generator script Miti provided at `2020T1_competition/enph353/enph353_gazebo/scripts`.

This modified script generates random plate configurations and then segments these plate images into individual characters. The output is a set of segmented characters "syntethically" generated. There is no data augmentation applied

The script takes one integer as an input argument, which specifies the amount of plates to randomly generate. The amount of output images will be 6 times the input integer ( as there are 6 characters for each plate).

#### Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ python_plate_generator.py 2
```
This example will produce 12 character images and save them in the `/generated_data/` folder.

