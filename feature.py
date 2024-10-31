import os
import math
import numpy as np
from easydev.progressbar import consoleprint
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from sympy.printing.numpy import const


# Function to extract differential entropy (DE) features from EEG data
def extract_features(filepath, subject_name):
    """
    Extract differential entropy (DE) features from EEG data.

    Args:
        filepath (str): Path to the EEG data file.
        subject_name (str): Name of the subject whose data is being processed.

    Returns:
        np.ndarray: Extracted DE features with shape (total_samples, 62, 5).
        np.ndarray: Corresponding labels for each sample.
    """
    # Load data from .mat file
    eeg_data = loadmat(filepath)
    sampling_rate = 200  # Hz, EEG data is downsampled to 200Hz

    all_features = []  # List to store all features from trials
    all_labels = []  # List to store all labels from trials
    labels = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0,
              3]  # Predefined labels for each trial

    for trial_index in range(24):
        # Extract trial data for the given subject
        trial_data = eeg_data[f'{subject_name}_eeg{trial_index + 1}']
        num_segments = len(trial_data[0]) // 100  # Calculate the number of 0.5-second segments
        trial_labels = [labels[trial_index]] * num_segments  # Create labels for each segment

        trial_features = []  # List to store features for the current trial
        for channel_index in range(62):
            signal = trial_data[channel_index]  # Extract signal for the current channel
            # Apply bandpass filters to extract different frequency bands
            bands = [
                butter_bandpass_filter(signal, 1, 4, sampling_rate),  # Delta band (1-4 Hz)
                butter_bandpass_filter(signal, 4, 8, sampling_rate),  # Theta band (4-8 Hz)
                butter_bandpass_filter(signal, 8, 14, sampling_rate),  # Alpha band (8-14 Hz)
                butter_bandpass_filter(signal, 14, 31, sampling_rate),  # Beta band (14-31 Hz)
                butter_bandpass_filter(signal, 31, 51, sampling_rate),  # Gamma band (31-51 Hz)
            ]
            # Compute DE features for each frequency band and each segment
            de_features = [
                [compute_de(band[segment * 100:(segment + 1) * 100]) for segment in range(num_segments)]
                for band in bands
            ]
            trial_features.append(de_features)  # Append DE features for the current channel
        trial_features = np.array(trial_features).transpose((2, 0, 1))  # Reshape to (num_segments, 62, 5)
        all_features.append(trial_features)  # Append features of the current trial to the list
        all_labels.extend(trial_labels)  # Append labels of the current trial to the list
        # Print progress to match the required output format
        # print(f'{trial_index + 1}-{num_segments}')

    # print(f'trial_DE shape: {np.vstack(all_features).shape}')
    return np.vstack(all_features), np.array(all_labels)  # Return all features and labels stacked together


# Function to design a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    """
    Design a Butterworth bandpass filter.

    Args:
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (float): Sampling rate.
        order (int): Filter order.

    Returns:
        tuple: Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyquist = 0.5 * fs  # Calculate Nyquist frequency
    low = lowcut / nyquist  # Normalize lower cutoff frequency
    high = highcut / nyquist  # Normalize upper cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Design bandpass filter
    return b, a  # Return filter coefficients


# Function to apply a Butterworth bandpass filter to the data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """
    Apply a Butterworth bandpass filter to the data.

    Args:
        data (np.ndarray): Input signal data.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (float): Sampling rate.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  # Get filter coefficients
    return lfilter(b, a, data)  # Apply filter to data


# Function to compute Differential Entropy (DE) of a signal segment
def compute_de(segment):
    """
    Compute Differential Entropy (DE) of a signal segment.

    Args:
        segment (np.ndarray): Signal segment.

    Returns:
        float: Differential entropy value.
    """
    variance = np.var(segment, ddof=1)  # Calculate variance of the segment
    return 0.5 * math.log(2 * math.pi * math.e * variance)  # Calculate DE using the variance


# Main script for extracting features from multiple subjects
input_directory = '../SEED-IV/eeg_raw_data/1'  # Directory containing the EEG data files
subject_files = [
    ('1_20160518', 'cz'),
    ('2_20150915', 'ha'),
    ('3_20150919', 'hql'),
    ('4_20151111', 'ldy'),
    ('5_20160406', 'ly'),
    ('6_20150507', 'mhw'),
    ('7_20150715', 'mz'),
    ('8_20151103', 'qyt'),
    ('9_20151028', 'rx'),
    ('10_20151014', 'tyc'),
    ('11_20150916', 'whh'),
    ('12_20150725', 'wll'),
    ('13_20151115', 'wq'),
    ('14_20151205', 'zjd'),
    ('15_20150508', 'zjy'),
]

X_all = []  # List to store features from all subjects
y_all = []  # List to store labels from all subjects

# Loop through each subject file and extract features
for subject_file, subject_name in subject_files:
    print(f'Processing {subject_file}...')  # Print the name of the file being processed
    features, labels = extract_features(os.path.join(input_directory, subject_file),
                                        subject_name)  # Extract features and labels
    X_all.append(features)  # Append extracted features to the list
    y_all.append(labels)

# Stack all features and labels into arrays
X_all = np.vstack(X_all)  # Stack features vertically
y_all = np.hstack(y_all)  # Stack labels horizontally

# Save the extracted features and labels to .npy files
np.save('./features/X_1D.npy', X_all)
np.save('./features/y.npy', y_all)

# Reshape features to 8x9 spatial grid
X_reshaped = np.zeros((len(y_all), 8, 9, 5))  # Initialize array to store reshaped features

# Map 62-channel features to an 8x9 grid according to electrode layout
channel_mapping = {
    (0, 2): 3,  # Mapping channel 3 to grid position (0, 2)
    (0, 3): 0, (0, 4): 1, (0, 5): 2,  # Mapping channels 0, 1, 2 to grid positions (0, 3), (0, 4), (0, 5)
    (0, 6): 4,  # Mapping channel 4 to grid position (0, 6)
    **{(i + 1, j): 5 + i * 9 + j for i in range(5) for j in range(9)},  # Mapping middle rows (1-5) to channels 5-49
    (6, 1): 50, (6, 2): 51, (6, 3): 52, (6, 4): 53, (6, 5): 54, (6, 6): 55, (6, 7): 56,
    # Mapping channels 50-56 to row 6
    (7, 2): 57, (7, 3): 58, (7, 4): 59, (7, 5): 60, (7, 6): 61,  # Mapping channels 57-61 to row 7
}

# Assign values to the reshaped feature array based on the channel mapping
for (row, col), channel in channel_mapping.items():
    X_reshaped[:, row, col, :] = X_all[:, channel, :]  # Assign channel data to corresponding grid position

# Save the reshaped features to a .npy file
np.save('./features/X89.npy', X_reshaped)
