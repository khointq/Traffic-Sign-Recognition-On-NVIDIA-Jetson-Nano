# Traffic-Sign-Recognition-On-NVIDIA-Jetson-Nano
A real-time traffic sign recognition system using YOLO NAS, optimized for deployment on NVIDIA Jetson Nano
## Overview 
The dataset comprises over 3,900 annotated images of 41 distinct Vietnamese traffic signs, collected under various traffic and weather scenarios. Training and fine-tuning were performed on Kaggle, and inference was optimized with NVIDIA TensorRT for deployment on a Jetson Nano B01 with limited resources. The final model achieves a mean Average Precision (mAP) above 90% at approximately 8.6 FPS on Jetson Nano, demonstrating its potential for real-time driver assistance and traffic monitoring applications, and serving as a foundation for future autonomous driving systems in Vietnam.
## Dependencies
### SuperGradients
### OpenCV
### Numpy
### TensorRT
### Pycuda
### Playsound
### Jetpack for JetsonNano
## Data
To label the images, the LabelImg image annotation tool is used. The images are then saved as VOCs and converted to COCO format to suit the model requirements. The figure below visualizes some of the images after being captioned.
<img width="1002" height="538" alt="image" src="https://github.com/user-attachments/assets/9e94a9ef-1b92-4b49-9ad9-76b5d240db3b" />
## Training on Kaggle
The training phase uses an NVIDIA T4 GPU, and the model is trained for 300 epochs with a batch size of 16. The source code used to train the model is available at training.ipynb
