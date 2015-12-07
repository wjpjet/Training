Instructions for running the dumper

Step 1:
Place positive training photos in positives/ and negatives in negatives/

Step 2:
The positives/ directory contains the groundtruth script creator which will label your
files and make a file called myground.gt
To run: $python3 scriptor.py

Step 3:
In the training directory run the dumper file to get the pixel intensity data
$python3 dumper.py positives --gt positives/myground.gt --nrands 9 --rotate-jitter ROTATE_JITTER

This will output an images.dump file.

Step 4:
rename images.dump to training_name.dump

Step 4:
Add negatives
$python3 dumper.py negatives/ 

Step 5:
Add negatives intensity data to positives
$cat images.dump >> training_name.dump

Step 6:
Time for training! If there is a lot of data you need to make sure you have enough RAM or else it will fail

Step 7:
basic training: ./picolrn training_name.dump <binary cascade output>

This should take a while. use nohup to put it in the background:
nohup ./picolrn training_name.dump <binary cascade output> &

Step 8:
cascade generation: ./picogen <path to trained cascade> 0.0 <detection function name> > <header file name>
