# CNN-Model-for-MNIST-Chinese-Handwritten-Numerals-
CNN Model for MNIST Chinese Handwritten Numerals 
Problem Introduction: In the field of machine learning, handwritten digit recognition is a classic problem. The MNIST Chinese Handwritten Digits dataset is a well-known dataset used to address this problem, containing images of handwritten digits. This report aims to describe the process of training a simple Convolutional Neural Network (CNN) model and a Residual Network (ResNet) model using the MNIST Chinese Handwritten Digits dataset.

Data Processing: The existing dataset is a binary file named chn_mnist. Based on the dataset usage instructions provided in the materials, it was determined that both the images field and the targets field are ndarray objects.

By examining the data format of a single image, it was observed that the images field contains 15,000 two-dimensional datasets of size 64×64, indicating that the data represents grayscale images.
![image](https://github.com/user-attachments/assets/c87e07e7-3dd0-4009-8f80-b0f319e906d1)

Translation:

Since the PyTorch framework requires training data to be structured as tensors and in three dimensions (our current data is a two-dimensional ndarray object, missing the depth dimension for color information, where 1 represents grayscale images and 3 represents RGB images), we performed type conversion and dimensionality expansion to meet this requirement.

Further Observation of the Target Field:
The output of the CNN's output layer is continuous, while the target values such as "hundred," "thousand," "ten thousand," and "hundred million" (represented as integers: 100, 1000, 10000, and 100000000) are clearly discontinuous. To correctly compare the output layer with the target values, the target data needs to be converted into continuous values. Here, "hundred" is mapped to 11, "thousand" to 12, "ten thousand" to 13, and "hundred million" to 14. This transformation is achieved by iterating through the dataset and modifying the target values.

CNN:
1. Is the Pooling Layer Necessary?
Although the training results appear to be good, the actual application performance is unsatisfactory.
It is worth considering whether important features are being lost during the pooling process. To address this, the pooling step was removed to test its impact on performance.

2. Hyperparameter Tuning
The key hyperparameters include the learning rate and the number of iterations:

A learning rate that is too large can result in poor model convergence.
A learning rate that is too small can significantly extend training time.
Proper adjustments were made to optimize these parameters.
3. Increasing the Number of Feature Extraction Layers
Initially, the number of extraction layers followed the standard MNIST recommendation (16 and 32 filters).
Considering that the resolution of this dataset's images is twice that of MNIST, the number of feature extraction layers was doubled (32 and 64 filters), while retaining the pooling layers.

![image](https://github.com/user-attachments/assets/4a96e085-e94a-455e-ab69-4c644947200b)

![image](https://github.com/user-attachments/assets/2aa519b6-0030-42ef-914b-5447cd4e12b1)

![image](https://github.com/user-attachments/assets/b8f90c8d-c4fd-4542-8389-94cf4f14099e)

RESNET:
Given that the depth and the number of feature extraction layers in the ResNet18 network are already sufficiently high, only the output layer was adjusted.

To address the high computational cost associated with the deep architecture of this model, CUDA hardware acceleration supported by PyTorch was utilized to accelerate training. The system leveraged GTX-1080 SLI, which consists of two GPUs.

Additionally, a relatively small learning rate was maintained to ensure that the model converges closer to the optimal solution.
![image](https://github.com/user-attachments/assets/1a7bdd2f-37b1-42ce-86c0-6a451f491216)

compare with CNN:
![image](https://github.com/user-attachments/assets/1f86a74a-a0cb-4842-8527-47686e942ff8)

CNN: 94.3%
ResNet_18: 99.2%

