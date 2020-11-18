# cnn_training

## perspective transform
perspective_transform_sprict.py is a script for taking in off angle images of license plates and flattening them. Try it out and let me know what you think! It seems to work ok for me so far. An example is as follows

Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ python perspective_transform.py simulation_data/weird_off_angle_9.png
```
## save rqt image
save_rqt_image.py is a script for saving the latest image published to "/R1/pi_camera/image_raw" and labelling it. This will be useful for tagging images for training the CNN. 

The syntax I have been using is:
`#Foldername/#PlateID_#Licensenumber`

The script checks how many file names match this format, increments the unique identifier and saves an image to:
`#Foldername/#PlateID_#Licensenumber_#uniqueintedifier`

Example:
``` bash
fizzer@enph353:~/ros_ws/src/cnn_training$ python save_rqt_image.py simulation_data/P2_EG31
```
