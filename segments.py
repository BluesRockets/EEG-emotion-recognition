import numpy as np

subjects = 15

# Load the preprocessed EEG data
X89_all_sessions = [np.load("./features/0_X89.npy"), np.load("./features/1_X89.npy"), np.load("./features/2_X89.npy")]
labels_all_sessions = [np.load("./features/0_labels.npy"), np.load("./features/1_labels.npy"), np.load("./features/2_labels.npy")]

for i in range(len(X89_all_sessions)):
    X89 = X89_all_sessions[i]
    labels = labels_all_sessions[i]
    # Define the image dimensions
    img_rows, img_cols, num_chan = 8, 9, 5
    falx = X89
    labels = labels[:int(labels.shape[0] / subjects)]

    # Reshape the data to (45 subjects, X89.shape[0] / subjects samples, img_rows, img_cols, num_chan)
    falx = falx.reshape((subjects, int(X89.shape[0] / subjects), img_rows, img_cols, num_chan))

    # Define the segment length
    segment_length = 6

    # Define the number of segments based on segment length
    # int(int(X89.shape[0] / subjects) / segment_length)
    num_segments = int(int(X89.shape[0] / subjects) / segment_length)
    # Iterate through different time intervals and assign corresponding labels
    # The labels are defined based on specific ranges within each subject's data
    label_intervals = []
    last_label = labels[0]
    for index, label in enumerate(labels):
        if label != last_label:
            label_intervals.append((label_intervals[-1][1] if len(label_intervals) > 0 else 0, index, last_label))
            last_label = label
    label_intervals.append((label_intervals[-1][1], len(labels) - 1, last_label))

    # Initialize the new dataset to store segmented data
    new_x = np.zeros((subjects, num_segments, segment_length, img_rows, img_cols, num_chan))
    new_y = np.array([])

    # Loop through each subject to segment the data
    for subject_idx in range(subjects):
        segment_idx = 0
        sample_idx = 0

        # Segment the data based on defined intervals and labels
        for start_idx, end_idx, label in label_intervals:
            sample_idx = start_idx
            while sample_idx + segment_length <= end_idx:
                # Extract segment of length `segment_length` from the data
                if falx[subject_idx, sample_idx:sample_idx + segment_length].shape[0] > 0:
                    new_x[subject_idx, segment_idx] = falx[subject_idx, sample_idx:sample_idx + segment_length]
                    new_y = np.append(new_y, label)
                sample_idx += segment_length
                segment_idx += 1

        new_x = new_x[:, :segment_idx, :, :, :, :]

        # Print the number of segments for each subject
        print(f'Subject {subject_idx}: {segment_idx} segments')

    # Save the segmented data and labels to .npy files
    np.save('./features/'+ str(i) +'_segmented_x_89.npy', new_x)
    np.save('./features/'+ str(i) +'_segmented_y_89.npy', new_y)