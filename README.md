# Transfer Learning X-Ray Classification for Pneumonia Detection

## Exploratory Data Analysis

### Training, Validation, and Test Set Counts

Comparison of training, validation, and test set counts shows that **~89%** of the images are in the **training set**. While only **0.27%** of the images are in **validation**, and **~11%** are in the **test set**.
For a dataset of 5856 images, which is quite small for image classification, the percent of images in the training, validation, and test set should be closer to **60%-20%-20%**.

[insert image](image_path)

- **Step 1: Randomly redistribute the images to 60:20:20**

### Class Distribution Among Sets

Comparison of the distribution of Pneumonia and Normal images between the training, validation, and test set shows that there is a significant imbalance between the groups.

[insert image](image_path)

The biggest issue is that the **validation set is not the same distribution as the test set**, which means that a model that performs well on the validation set may not do well on the test set.
Through the random redistribution of images in Step1, the image class distribution should be very similar between training, validation, and test sets after that step.
While the **class imbalance of Pneumonia to Normal is significant**, it can be accounted for by **applying weights to the training** to penalize the model more for wrong of the less-represented class (Normal).
This should mitigate the class imbalance issue, since it is not a severe imbalance.

[insert image](image_path)

- **Step 2: Apply weights to the classes for training** 

### Pixel Intensities Between Pneumonia and Normal


