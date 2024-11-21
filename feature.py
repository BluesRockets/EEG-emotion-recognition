import os
import math
import numpy as np
from easydev.progressbar import consoleprint
from scipy.signal import butter, lfilter
from scipy.io import loadmat

data_directory = '..\\SEED-IV\\eeg_raw_data'  # Directory containing the EEG data files
data_labels = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
               [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
               [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]


# Function to extract differential entropy (DE) features from EEG data
def extract_features(filepath, subject_name, session_index):
    # Load data from .mat file
    eeg_data = loadmat(filepath)
    sampling_rate = 200  # Hz, EEG data is downsampled to 200Hz

    all_features = []  # List to store all features from trials
    all_labels = []  # List to store all labels from trials

    for trial_index in range(24):
        # Extract trial data for the given subject
        trial_data = eeg_data[f'{subject_name}_eeg{trial_index + 1}']
        num_segments = len(trial_data[0]) // 100  # Calculate the number of 0.5-second segments
        trial_labels = [data_labels[session_index][trial_index]] * num_segments  # Create labels for each segment

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
    nyquist = 0.5 * fs  # Calculate Nyquist frequency
    low = lowcut / nyquist  # Normalize lower cutoff frequency
    high = highcut / nyquist  # Normalize upper cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Design bandpass filter
    return b, a  # Return filter coefficients


# Function to apply a Butterworth bandpass filter to the data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  # Get filter coefficients
    return lfilter(b, a, data)  # Apply filter to data


# Function to compute Differential Entropy (DE) of a signal segment
def compute_de(segment):
    variance = np.var(segment, ddof=1)  # Calculate variance of the segment
    return 0.5 * math.log(2 * math.pi * math.e * variance)  # Calculate DE using the variance


# extract file name and participant's name
subject_files = []
for session_dir in os.listdir(data_directory):
    if not session_dir.startswith('.'):
        files = []
        child_files = os.listdir(os.path.join(data_directory, session_dir))
        for child_file_name in child_files:
            if not child_file_name.startswith('.'):
                child_file_path = os.path.join(data_directory, session_dir, child_file_name)
                if os.path.isfile(child_file_path):
                    eeg_data = loadmat(child_file_path)
                    keys = eeg_data.keys()
                    filtered_keys = [key for key in keys if isinstance(key, str) and not key.startswith('__')]
                    if len(filtered_keys) > 0:
                        files.append((child_file_path, filtered_keys[0].split('_')[0]))
        files = sorted(files)
        subject_files.append(files)

X_all = []  # List to store features from all subjects
y_all = []  # List to store labels from all subjects

# Loop through each subject file and extract features
for session_index in range(len(subject_files)):
    session_features = []
    session_labels = []
    for file_path, subject_name in subject_files[session_index]:
        print(f'Processing {file_path}...')
        features, labels = extract_features(file_path, subject_name, session_index)
        session_features.append(features)
        session_labels.append(labels)
    X_all.append(np.concatenate(session_features, axis=0))
    y_all.append(np.concatenate(session_labels, axis=0))

for i in range(len(X_all)):
    np.save('./features/'+ str(i) + '_labels.npy', y_all[i])

    # Reshape features to 8x9 spatial grid
    X_reshaped = np.zeros((len(X_all[i]), 8, 9, 5))  # Initialize array to store reshaped features

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
        X_reshaped[:, row, col, :] = X_all[i][:, channel, :]  # Assign channel data to corresponding grid position

    # Save the reshaped features to a .npy file
    np.save('./features/'+ str(i) + '_X89.npy', X_reshaped)
