"""
Splits a dataset and corresponding labels into training and testing datasets with equal distribution among classes
param:
    @data: data array
    @lbls: labels array (should correspond to data array)
    @test_size: desired size of test dataset (0-1)
returns:
    train_data, train_lbls, test_data, test_lbls
"""
def train_test_split(data, lbls, test_size):
    # Determine unique set of classes
    classes = list(set(lbls))
    class_count = len(classes)

    # Determine the amount of samples required per class to satisfy the split,
    # assuming all classes have equal distribution in the original dataset
    sample_count, data_features = data.shape
    class_sample_count = int(sample_count / class_count)
    class_test_sample_count = round(class_sample_count*test_size)
    class_train_sample_count = class_sample_count-class_test_sample_count

    # Initialize data arrays
    total_test_sample_count = class_test_sample_count*class_count
    total_train_sample_count = class_train_sample_count*class_count
    train_data = np.zeros((total_train_sample_count,data_features))
    train_lbls = np.zeros(total_train_sample_count,dtype=np.uint8)
    test_data = np.zeros((total_test_sample_count,data_features))
    test_lbls = np.zeros(total_test_sample_count, dtype=np.uint8)
    # Iterate classes and split the samples related to each class into training and test datasets
    for i, label in enumerate(classes):
        # Filter samples by current class
        label_samples = np.array([x for i, x in enumerate(data) if lbls[i] == label])
        # Randomize training and test indices within class data set
        train_indices = sample(range(0,class_sample_count), class_train_sample_count)
        test_indices = list(set(range(class_sample_count)) - set(train_indices))
        # Assign randomized data samples to training and test arrays in proportion with the desired test size
        train_data[i*class_train_sample_count:(i+1)*class_train_sample_count] = label_samples[train_indices]
        train_lbls[i*class_train_sample_count:(i+1)*class_train_sample_count] = [label for i in range(class_train_sample_count)]
        test_data[i*class_test_sample_count:(i+1)*class_test_sample_count] = label_samples[test_indices]
        test_lbls[i*class_test_sample_count:(i+1)*class_test_sample_count] = [label for i in range(class_test_sample_count)]

    # Shuffle training and testing datasets to randomize order
    train_data, train_lbls = shuffle(train_data, train_lbls)
    test_data, test_lbls = shuffle(test_data, test_lbls)

    return train_data, test_data, train_lbls, test_lbls