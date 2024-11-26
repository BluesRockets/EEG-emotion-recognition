# EEG Emotion Recognition with CNN
This project implements a Convolutional Neural Network (CNN) for emotion recognition based on EEG data. The following instructions will guide you through the setup and usage of the project.

## Requirements
- Python 3.x
- Numpy
- Scipy
- PyTorch
- sklearn

## Usage
### 1. Modify the data_directory Variable:
In feature.py, set the data_directory variable to point to your local folder containing the raw EEG data files.
The data must be sourced directly from the SEED-V dataset.
### 2. Create Features Directory:
In the project directory, create a folder named features.
### 3. Run Feature Extraction:
Execute feature.py. This will process the EEG data and generate two files in the features directory:
X89.npy: This file contains the processed EEG data.
labels.npy: This file contains the corresponding labels for the EEG data.
### 4. Segment the Data:
Run segments.py to segment the processed EEG data. This will create the following files in the features directory:
segmented_x_89.npy: Segmented EEG data.
segmented_y_89.npy: Corresponding labels for the segmented data.
### 5. Train the Model:
Execute train.py to start training the CNN model. After training, the model parameters will be saved in the result folder as name of model.pt.
### 6. Evaluate the Model:
Finally, run evaluation.py to test the model's prediction accuracy.