from sklearn.decomposition import PCA
from mnist import MNIST
import numpy as np
import scipy.io as scio
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from random import sample

""" 
Load MNIST dataset from directory
param:
    @fp: path to directory containing data
returns:
    train_images, train_lbls, test_images, test_lbls
"""
def loadMNIST(fp):
    mndata = MNIST(fp, return_type='numpy')
    train_images, train_lbls = mndata.load_training()
    test_images, test_lbls = mndata.load_testing()
    return train_images, train_lbls, test_images, test_lbls


""" 
Load ORL dataset from directory and splits it training and testing datasets
param:
    @fp: path to directory containing data
    @test_size: size of test dataset (0-1)
returns:
    train_data, train_lbls, test_data, test_lbls
"""
def loadORL(fp, test_size=0.3):
    data = np.array(scio.loadmat(fp + '/orl_data.mat')['data']).transpose()
    lbls = np.array(scio.loadmat(fp + '/orl_lbls.mat')['lbls']).ravel()
    # Split data into training and testing datasets
    train_data, test_data, train_lbls, test_lbls = train_test_split(data, lbls, test_size=test_size)

    return train_data, train_lbls, test_data, test_lbls


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
        # Assign randomized data to training and test arrays
        train_data[i*class_train_sample_count:(i+1)*class_train_sample_count] = label_samples[train_indices]
        train_lbls[i*class_train_sample_count:(i+1)*class_train_sample_count] = [label for i in range(class_train_sample_count)]
        test_data[i*class_test_sample_count:(i+1)*class_test_sample_count] = label_samples[test_indices]
        test_lbls[i*class_test_sample_count:(i+1)*class_test_sample_count] = [label for i in range(class_test_sample_count)]

    # Shuffle training and testing datasets to randomize order
    train_data, train_lbls = shuffle(train_data, train_lbls)
    test_data, test_lbls = shuffle(test_data, test_lbls)

    return train_data, test_data, train_lbls, test_lbls


""" 
Apply PCA to training and testing data samples
param:
    @train_data: training data
    @test_data: testing data
returns:
    train_images, train_lbls, test_images, test_lbls
"""
def pca(train_data, test_data):
    # Initialize training PCA to 2-dimensions
    pca = PCA(n_components=2)
    pca.fit(train_data)
    # Fit PCA to data and transform
    pca_train_data = pca.transform(train_data)
    pca_test_data = pca.transform(test_data)
    return pca_train_data, pca_test_data


""" 
Plots the class mean vectors of supplied MNIST data
param:
    @data: MNIST data
    @labels: data labels
    @tile: plot title
"""
def plot_mnist_centroids(data, labels, title=""):
    # Create set of classes in data set
    classes = list(set(labels))

    # Calculate mean vector of each class
    clf = NearestCentroid()
    clf.fit(data, labels)
    centroids = clf.centroids_

    # https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot
    plt.figure()
    plt.suptitle(title, fontsize=14)
    for i, class_center in enumerate(centroids):
        pixels = np.array(class_center, dtype='uint8')

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))

        # Plot each mean vector as a gray scale image in a subplot
        plt.subplot(2,5,i+1)
        plt.title('Label: {label}'.format(label=classes[i]))
        plt.imshow(pixels, cmap='gray')

    plt.draw()


""" 
Plots the class mean vectors of supplied ORL data
param:
    @data: ORL data
    @labels: data labels
    @tile: plot title
"""
def plot_orl_centroids(data, labels, title=""):
    # Create set of classes in data set
    classes = list(set(labels))

    # Calculate mean vector of each class
    clf = NearestCentroid()
    clf.fit(data, labels)
    centroids = clf.centroids_

    plt.figure(figsize=(18,12))
    plt.suptitle(title, fontsize=14)
    for i, class_center in enumerate(centroids):
        pixels = np.array(class_center, dtype='float')

        # Reshape the array into 40 x 30 array (2-dimensional array)
        pixels = pixels.reshape((30, 40)).transpose()   # image vectors are sideways

        # Plot each mean vector as a gray scale image in a subplot
        plt.subplot(4,10,i+1)
        plt.title('Label: {label}'.format(label=classes[i]))
        plt.imshow(pixels, cmap='gray')

    plt.draw()


""" 
Scatterplots color coded 2D data points with unique color code for each class
param:
    @data: 2D data
    @labels: list of data labels
    @tile: plot title
"""
def plot_2D_data(data, labels, title=""):
    # Create set of classes in data set
    classes = list(set(labels))
    class_count = len(classes)

    # Generate color map with unique color for each class
    color_map = plt.get_cmap('rainbow')
    colors = color_map(np.linspace(0, 1.0, class_count))

    # Generate scatter plots for each class
    plots = []
    plt.figure()
    plt.title(title)
    for i, label in enumerate(classes):
        # Group data into numpy arrays for each class
        class_data = np.asarray([x for j, x in enumerate(data) if labels[j]==label])
        x = class_data[:,0]
        y = class_data[:,1]
        plots.append(plt.scatter(x,y,color=colors[i]))

    # Add legend
    plt.legend(plots,
               classes,
               scatterpoints=1,
               loc='upper right',
               ncol=2,
               fontsize=8)

    plt.draw()

""" 
Scatterplots color coded 2D data points with unique color code for each class in a subplot for each list of labels supplied
param:
    @data: 2D data
    @labels: list of data labels 
    @tile: plot title
    @subplot_titles: list of titles of subplots
"""
def subplot_2D_data(data, dataset_labels, title="", subplot_titles=[]):
    plt.figure()
    plt.suptitle(title)
    for i,labels in enumerate(dataset_labels):
        plt.subplot(len(dataset_labels), 1, i + 1)
        # Create set of classes in data set
        classes = list(set(labels))
        class_count = len(classes)

        # Generate color map with unique color for each class
        color_map = plt.get_cmap('rainbow')
        colors = color_map(np.linspace(0, 1.0, class_count))

        # Generate scatter plots for each class
        plots = []
        if subplot_titles != []:
            plt.title(subplot_titles[i])
        for i, label in enumerate(classes):
            # Group data into numpy arrays for each class
            class_data = np.asarray([x for j, x in enumerate(data) if labels[j] == label])
            x = class_data[:, 0]
            y = class_data[:, 1]
            plots.append(plt.scatter(x, y, color=colors[i]))

        # Add legend
        plt.legend(plots,
                   classes,
                   scatterpoints=1,
                   loc='upper right',
                   ncol=2,
                   fontsize=8)

    plt.draw()