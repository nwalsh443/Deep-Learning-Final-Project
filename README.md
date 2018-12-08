## Deep Learning Holiday Image Manipulation
Created by Noah Walsh, Ben Valois, Rick Djeuhon, Derek Windahl "VerseForty", and Jake Hamilton

Our project manipulates a non-holiday image to appear more like a holiday with the deep convolutional neural network Inception v3, that we trained with a custom holiday classification image dataset.

## How to use it
Clone or download this project. Then log onto Noah Walsh's Dev Cloud Account through PuTTY. Upload the Holiday_images training folder. Make sure to rename or delete any existing '.h5' weight files, such as holiday_weights.h5, or '.png' output files, such as result.png, before beginning. Enter these commands:

qsub -I
source activate testEnv
ipython

Then copy and paste the code from the TrainHolidayV3.py file into the ipython shell, making sure the paths to the Holiday_images training folder are correct. Then run the code. When it is done, the Inception network weights file holiday_weights.h5 should be saved.

Then copy and paste the code from the HolidayImageManipulator.py file into the ipython shell. Make sure the path to the input image you are manipulating is correct, and that the path to the output file result.png that will be created when the code is done is correct as well. Run the code. When done, the output file result.png should be saved. If the output image is completely black, that means your weights file is corrupted, and you need to delete it and the output image, and run the TrainHolidayV3.py code again, and then run the HolidayImageManipulator.py code again. 
