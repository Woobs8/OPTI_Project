from sklearn.decomposition import PCA
from mnist import MNIST
import numpy as np
import scipy.io as scio
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import itertools

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
def loadORL(fp, test_size=0.3, seed=None):
    data = np.array(scio.loadmat(fp + '/orl_data.mat')['data']).transpose()
    lbls = np.array(scio.loadmat(fp + '/orl_lbls.mat')['lbls']).ravel()
    # Split data into training and testing datasets
    if seed != None:
        train_data, test_data, train_lbls, test_lbls = train_test_split(data, lbls, test_size=test_size,stratify=lbls, random_state=seed)
    else:
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
Apply t-SNE to training and testing data samples
param:
    @train_data: training data
    @test_data: testing data
returns:
    train_images, train_lbls, test_images, test_lbls
"""
def tsne(train_data, test_data, n_components=2):
    # Speed ud TSNE computations by performing initial dimensionality reduction
    pca = PCA(n_components=50)
    pca.fit(train_data)
    pca_train_data = pca.transform(train_data)
    pca_test_data = pca.transform(test_data)

    # Initialize training PCA to 2-dimensions
    tsne = TSNE(n_components=n_components)
    tsne_train_data = tsne.fit_transform(pca_train_data)
    tsne_test_data = tsne.fit_transform(pca_test_data)
    return tsne_train_data, tsne_test_data


""" 
Plots the class mean vectors of supplied MNIST data
param:
    @data: MNIST data
    @labels: data labels
    @tile: plot title
    @fp: path to store file in
"""
def plot_mnist_centroids(data, labels, title="", fp=""):
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
    if fp != "":
        plt.savefig(fp)
    plt.draw()


""" 
Plots the class mean vectors of supplied ORL data
param:
    @data: ORL data
    @labels: data labels
    @tile: plot title
    @fp: path to store file in
"""
def plot_orl_centroids(data, labels, title="", fp=""):
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
        plt.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    if fp != "":
        plt.savefig(fp)
    plt.draw()


""" 
Scatterplots color coded 2D data points with unique color code for each class
param:
    @data: 2D data
    @labels: list of data labels
    @tile: plot title
    @fp: path to store file in
"""
def plot_2D_data(data, labels, title="", fp=""):
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

        # Reduce marker size if number of samples is high
        if len(data) > 1000:
            plots.append(plt.scatter(x,y,color=colors[i],s=0.5))
        else:
            plots.append(plt.scatter(x, y, color=colors[i]))

    # Add legend
    lgnd = plt.legend(plots,
               classes,
               scatterpoints=1,
               loc='upper right',
               ncol=2,
               fontsize=8)

    for handle in lgnd.legendHandles:
        handle._sizes = [5]

    if fp != "":
        plt.savefig(fp)
    plt.draw()


""" 
Scatterplots color coded 2D data points with unique color code for each class in a subplot for each list of labels supplied
param:
    @data: 2D data
    @labels: list of data labels 
    @tile: plot title
    @subplot_titles: list of titles of subplots
    @fp: path to store file in
"""
def subplot_2D_data(data, dataset_labels, title="", subplot_titles=[],  fp=""):
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

            # Reduce marker size if number of samples is high
            if len(data) > 1000:
                plots.append(plt.scatter(x, y, color=colors[i], s=0.5))
            else:
                plots.append(plt.scatter(x, y, color=colors[i]))

        # Add legend
        lgnd = plt.legend(plots,
                   classes,
                   scatterpoints=1,
                   loc='upper right',
                   ncol=2,
                   fontsize=8)

        for handle in lgnd.legendHandles:
            handle._sizes = [5]

    if fp != "":
        plt.savefig(fp)
    plt.draw()


""" 
Plots a confusion matrix for the classified and labeled data
param:
    @pred_labels: classified data labels
    @true_labels: actual data labels
    @normalize: normalize data if True
    @tile: plot title
    @fp: path to store file in
"""
def plot_confusion_matrix(pred_labels, true_labels, normalize=False, title="", fp=""):
    classes = np.unique(true_labels)
    conf_mat = confusion_matrix(pred_labels, true_labels)
    np.set_printoptions(precision=2)
    cmap = plt.cm.Blues

    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=8)

    # Only plot text for small confusion matrices
    if len(classes) <= 10:
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mat.max() / 2.
        for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            plt.text(j, i, format(conf_mat[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    np.set_printoptions()
    if fp != "":
        plt.savefig(fp)
    plt.draw()


""" 
Plots the decision boundary of the @clf classifer function, and overlays a scatter plot of the testing data
param:
    @clf: classifier function
    @train_data: training data
    @train_lbls: training labels
    @test_data: testing data
    @test_lbls: testing labels
    @tile: plot title
    @fp: path to store file in
"""
def plot_decision_boundary(clf, test_data, test_lbls, title, fp="", *args):
    # Generate figure and color map
    color_map = plt.cm.RdBu
    plt.figure()
    plt.title(title)

    # Separate data into features and construct meshgrid
    X0_test, X1_test = test_data[:, 0], test_data[:, 1]
    x_min, x_max = X0_test.min() - 1, X0_test.max() + 1
    y_min, y_max = X1_test.min() - 1, X1_test.max() + 1
    xx_test, yy_test = np.meshgrid(np.arange(x_min, x_max),
                         np.arange(y_min, y_max))

    # Classify testing data using @clf classifier function
    Z, score = clf.predict(np.c_[xx_test.ravel(), yy_test.ravel()], test_lbls)
    classes = np.unique(Z)
    Z = Z.reshape(xx_test.shape)    # Reshape into meshgrid shape

    # Plot decision boundary with testing data
    plt.contourf(xx_test, yy_test, Z, len(classes)+1, cmap=color_map, alpha=0.8)
    plt.scatter(X0_test, X1_test, c=test_lbls, cmap=color_map, s=5, edgecolors='k', lw=0.5)
    plt.xlim(xx_test.min(), xx_test.max())
    plt.ylim(yy_test.min(), yy_test.max())
    plt.xticks(())
    plt.yticks(())

    if fp != "":
        plt.savefig(fp)
    plt.draw()