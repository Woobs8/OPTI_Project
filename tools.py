from sklearn.decomposition import PCA
from mnist import MNIST
import numpy as np
import scipy.io as scio
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    train_data, test_data, train_lbls, test_lbls = train_test_split(data, lbls, test_size=test_size,stratify=lbls)

    return train_data, train_lbls, test_data, test_lbls


""" 
Apply PCA to training and testing data samples
param:
    @train_data: training data
    @test_data: testing data
returns:
    train_images, train_lbls, test_images, test_lbls
"""
def pca(train_data, test_data, n_components=2):
    # Initialize training PCA to 2-dimensions
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
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