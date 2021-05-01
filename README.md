# Mask Detection Webcam App - Python
 Image processing with the use of VGG algorithm, which is a complex structure that includes CNNs
 
 
Prerequisties:
pip install opencv-python

pip install numpy 

pip install pandas 

Pip install tensorflow

Pip install keras

Pip install h5py



This project is created for VA455 course by me. VA 455 is Physical Computing course and it is like unification of both Computer Science and Visual Arts. 

The main skeleton of the project is taken from the https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/ project and it is converted to mask detection program with including it the OpenCV library.

There are two python files in the repo, va455_finalize.py and va455_model.py. 

va455_model.py is for creating the decison-maker machine learning model which is going to be used in the va455_finalize.py. Thus va455_model.py should be run first and after this run, the model is going to be created at the same folder, with the name of final_model.h7 . As I said, this model will be used by va455_finalize.py, to determine if the person who exists in the taken photo wears a mask or not.

Whenever va455_finalize.py started to run, webcam is being executed and program asks user to press a key in the keyboard, in order to take that moment's photo from webcam.

![resim](https://user-images.githubusercontent.com/55497058/116768571-288ead00-aa40-11eb-97a2-573d34ab63df.png)

![resim](https://user-images.githubusercontent.com/55497058/116768578-33494200-aa40-11eb-8ad7-330dd01f3953.png)



After user presses the key, webcam is being closed and the taken photo is being processed by the program. The following pictures will be shown as output, dependent on whether user wears mask or not.

![resim](https://user-images.githubusercontent.com/55497058/116768489-a8684780-aa3f-11eb-9907-8797181d9501.png)

![resim](https://user-images.githubusercontent.com/55497058/116768520-d51c5f00-aa3f-11eb-85f9-fb3ffb8deb3b.png)


Note: Rarely, when the enviroment is not shiny and clear or when program forced to evaluate non-human head objects or pictures, it may return negative floats and it may contain 1 which can be led false printing statements which only happens at the mentioned situations. I did not fixed it when I had finished it and I am just uploading it here for storage issues, So if it is being used in somewhere, better to handle this possible problem. Also when I did this project, it was 2020 april and thus there were not that many datasets with humans wearing masks, thus the dataset is relatively small and would be better to extend it before use.


Some deeper informations about VGG which taken from Internet exists below for clarification.

•	Input. VGG takes in a 224x224 pixel RGB image. For the ImageNet competition, the authors cropped out the center 224x224 patch in each image to keep the input image size consistent.
•	Convolutional Layers. The convolutional layers in VGG use a very small receptive field (3x3, the smallest possible size that still captures left/right and up/down). There are also 1x1 convolution filters which act as a linear transformation of the input, which is followed by a ReLU unit. The convolution stride is fixed to 1 pixel so that the spatial resolution is preserved after convolution.
•	Fully-Connected Layers. VGG has three fully-connected layers: the first two have 4096 channels each and the third has 1000 channels, 1 for each class.
•	Hidden Layers. All of VGG’s hidden layers use ReLU (a huge innovation from AlexNet that cut training time). VGG does not generally use Local Response Normalization (LRN), as LRN increases memory consumption and training time with no particular increase in accuracy.
The below one is an infographic picture about the VGG structure.

![resim](https://user-images.githubusercontent.com/55497058/116768544-fb41ff00-aa3f-11eb-9389-67aacbb60760.png)


