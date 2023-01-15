# Transfer Learning X-Ray Classification for Pneumonia Detection

## Exploratory Data Analysis
From the notebook [transfer-learning-x-ray-eda](./notebooks/transfer-learning-x-ray-eda.ipynb)

### Training, Validation, and Test Set Counts

Comparison of training, validation, and test set counts shows that **~89%** of the images are in the **training set**. While only **0.27%** of the images are in **validation**, and **~11%** are in the **test set**.
For a dataset of 5856 images, which is quite small for image classification, the percent of images in the training, validation, and test set should be closer to **60%-20%-20%**.

![File Counts](./images/File_counts.png)

- **Step 1: Randomly redistribute the images to 60:20:20**

### Class Distribution Among Sets

Comparison of the distribution of Pneumonia and Normal images between the training, validation, and test set shows that there is a significant imbalance between the groups.

![Class Distribtuions](./images/Class_distributions.png)

The biggest issue is that the **validation set is not the same distribution as the test set**, which means that a model that performs well on the validation set may not do well on the test set.
Through the random redistribution of images in Step1, the image class distribution should be very similar between training, validation, and test sets after that step.
While the **class imbalance of Pneumonia to Normal is significant**, it can be accounted for by **applying weights to the training** to penalize the model more for wrong of the less-represented class (Normal).
This should mitigate the class imbalance issue, since it is not a severe imbalance.

- **Step 2: Apply weights to the classes for training** 

### Pixel Intensities Between Pneumonia and Normal

It becomes clear from the pixel intensities below that there can be a **significant difference based on how bright the X-ray is**. Since these are from real X-rays, it can be assumed that **the model will have to perform well on dark and bright X-rays**. The best scenario is to make sure the distribution of images is very well shuffled and distributed proportionally across all sets of the groups.

- **Make sure the redistribution of images is random to have a mix of pixel intensities in each set**

![pixels_1](./images/pixels_1.png)
![pixels_2](./images/pixels_2.png)
![pixels_3](./images/pixels_3.png)
![pixels_4](./images/pixels_4.png)
![pixels_5](./images/pixels_5.png)
![pixels_6](./images/pixels_6.png)

## Choosing a Transfer Model

Compare base models of VGG-16, InceptionV3, ResNet50, and DenseNet201 for performance. Use the best performing model for fine tuning.

### Pre-Processing




