## DeepHoliday: Deep Learning Holiday Image Manipulation
Created by Noah Walsh, Ben Valois, Rick Djeuhon, Derek Windahl "VerseForty", and Jake Hamilton

DeepHoliday manipulates non-holiday images with the deep convolutional neural network classifier Inception V3, trained on a custom dataset of holiday images, to make the non-holiday input images look more holiday-themed and festive.

## Environment Package Requirements
tensorflow 1.5

keras

scipy

pandas

numpy

pillow

## How to use it
Clone or download this project. Then log onto Noah Walsh's Dev Cloud Account through PuTTY. The Holiday_images training folder is already on Noah's Dev Cloud Account, but you can also download it from the shared Google Drive, Data for Deep Learning Holiday Image Manipulation Project, and upload it to the Dev Cloud account if you want. The .jpg input images are also on Dev Cloud, but you can download them and upload them to Dev Cloud again. Make sure to rename or delete any existing '.h5' weight files, such as holiday_weights.h5, or '.png' output files, such as result.png, before beginning. Enter these commands:

qsub -I

source activate testEnv

ipython

Then copy and paste the code from the TrainHolidayV3.py file into the ipython shell, making sure the paths to the Holiday_images training folder are correct. Make sure that the train and val folders have exactly 1000 folders in each of them. Then run the code. When it is done, the Inception network weights file holiday_weights.h5 will be saved.

Then copy and paste the code from the HolidayImageManipulator.py file into the ipython shell. Make sure the path to the input image you are manipulating is correct, and that the path to the output file result.png that will be created when the code is done is correct as well. Run the code. When done, the output file result.png should be saved. If the output image is completely black, that means your weights file is corrupted, and you need to delete it and the output image, and run the TrainHolidayV3.py code again, and then run the HolidayImageManipulator.py code again. 
