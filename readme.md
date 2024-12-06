# EEG Emotion Recognition with CNN
This project implements a Convolutional Neural Network (CNN) for emotion recognition based on EEG data. The following instructions will guide you through the setup and usage of the project.

## Requirements
- Python 3.x
- Numpy
- Scipy
- PyTorch
- sklearn

## Structure
```
├── EDA/                 # Directory for Exploratory Data Analysis scripts  
├── feature/             # Directory for storing preprocessed data to read by the modal
├── model_states/        # Directory for saving trained modal's parameters
│   └── model.pt         # File for saving the CNN-LSTM model's weights
├── results/             # Directory for saving training process output in different times
├── evaluation.py        # Script for evaluating model performance on test data
├── README.md            # Project documentation file
├── feature.py           # Script containing feature extraction methods
├── segments.py          # Additional training and experiment-related scripts
└── train.py             # Script defining the general training loop
```

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

## Model Architecture
Convolutional Neural Network (CNN)
Defined in train.py by the Class ConvNet, this Class defined all the layers and all parameters of CNN network.

Long Short-Term Memory (LSTM)
Defined in train.py by the Class EEGNet, this Class defined all the layers and all parameters of LSTM.

## Training the Model
Training process is in the file train.py, which read preprocessed data and extracted features from folder features/.
Then saving the all parameters of the trained model in to the file model_states/model.pt, in order to evaluation the model many times without training again.

## Evaluation
The evaluation steps are embedded in the evaluation.py script, which reports:
- The value of loss function changes
- Validation accuracy
- Total time consumption
