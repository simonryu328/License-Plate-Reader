#!/bin/bash   
echo "First arg: $0"
echo "First arg: $1"
echo "path *"
for f in ./"$1"*.png; do 
	python perspective_transform_script.py "$f"; 
done

