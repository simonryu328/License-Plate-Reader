#!/bin/bash   
for f in ./"$1"*.png; do 
	python segment_license_plate_images.py "$f"; 
done
