import numpy as np

subjects = 15

# Load the preprocessed EEG data
X89 = np.load("./features/X89.npy")

# Define the image dimensions
img_rows, img_cols, num_chan = 8, 9, 5
falx = X89



# Reshape the data to (45 subjects, 6870 samples, img_rows, img_cols, num_chan)
falx = falx.reshape((subjects, 6870, img_rows, img_cols, num_chan))

# Define the segment length
segment_length = 6

# Define the number of segments based on segment length
num_segments = int(6870 / segment_length)

# Initialize the new dataset to store segmented data
new_x = np.zeros((subjects, num_segments, segment_length, img_rows, img_cols, num_chan))
new_y = np.array([])

# Loop through each subject to segment the data
for subject_idx in range(subjects):
    segment_idx = 0
    sample_idx = 0

    # Iterate through different time intervals and assign corresponding labels
    # The labels are defined based on specific ranges within each subject's data
    label_intervals = [
        (0, 336, 1), (336, 526, 2), (526, 924, 3), (924, 1184, 0),
        (1184, 1360, 2), (1360, 1684, 0), (1684, 1990, 0), (1990, 2408, 1),
        (2408, 2698, 0), (2698, 3036, 1), (3036, 3136, 2), (3136, 3356, 1),
        (3356, 3790, 1), (3790, 4128, 1), (4128, 4646, 2), (4646, 4928, 3),
        (4928, 5064, 2), (5064, 5422, 2), (5422, 5702, 3), (5702, 5798, 3),
        (5798, 6022, 0), (6022, 6246, 3), (6246, 6596, 0), (6596, 6870, 3),
    ]

    # Segment the data based on defined intervals and labels
    for start_idx, end_idx, label in label_intervals:
        sample_idx = start_idx
        while sample_idx + segment_length <= end_idx:
            # Extract segment of length `segment_length` from the data
            new_x[subject_idx, segment_idx] = falx[subject_idx, sample_idx:sample_idx + segment_length]
            # Append corresponding label to the label array
            new_y = np.append(new_y, label)
            # Move to the next segment
            sample_idx += segment_length
            segment_idx += 1

    # Print the number of segments for each subject
    print(f'Subject {subject_idx}: {segment_idx} segments')

# Save the segmented data and labels to .npy files
np.save('./features/' + str(segment_length) + 'x_89.npy', new_x)
np.save('./features/' + str(segment_length) + 'y_89.npy', new_y)
# The shape of `new_y` is expected to be (45 * num_segments,) depending on the segment length.