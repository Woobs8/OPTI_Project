import os
from tools import loadMNIST, loadORL, pca
from classification import nc, nsc


""" ********* Loading MNIST samples ********* """
# MNIST sample directory
MNIST_DIR = "MNIST"
MNIST_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + MNIST_DIR
# Load MNIST samples
mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls = loadMNIST(MNIST_PATH)
# Apply PCA to MNIST samples
pca_mnist_train_images, pca_mnist_test_images = pca(mnist_train_images, mnist_test_images)


""" ********* Loading ORL samples ********* """
# ORL sample directory
ORL_DIR = "ORL"
ORL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + ORL_DIR
# Load ORL samples
orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls = loadORL(ORL_PATH)
# Apply PCA to ORL samples
pca_orl_train_images, pca_orl_test_images = pca(orl_train_images, orl_test_images)


""" ********* Classifying MNIST samples ********* """
# Nearest Centroid
nc_mnist_class, nc_mnist_score = nc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls)
# Nearest Centroid w/ PCA data
pca_nc_mnist_class, pca_nc_mnist_score = nc(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images, mnist_test_lbls)
# Nearest Subclass Centroid (2 subclasses)
nsc_2_mnist_class, nsc_2_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,2)
# Nearest Subclass Centroid (3 subclasses)
nsc_3_mnist_class, nsc_3_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,3)
# Nearest Subclass Centroid (5 subclasses)
nsc_5_mnist_class, nsc_5_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,5)


""" ********* Classifying ORL samples ********* """
# Nearest Centroid
nc_orl_class, nc_orl_score = nc(orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls)
# Nearest Centroid w/ PCA data
pca_nc_orl_class, pca_nc_orl_score = nc(pca_orl_train_images, orl_train_lbls, pca_orl_test_images, orl_test_lbls)
# Nearest Subclass Centroid (2 subclasses)
nsc_2_orl_class, nsc_2_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,2)
# Nearest Subclass Centroid (3 subclasses)
nsc_3_orl_class, nsc_3_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,3)
# Nearest Subclass Centroid (5 subclasses)
#nsc_5_orl_class, nsc_5_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,5)


""" ********* Classification scores ********* """
print("*** Classification Scores ***")
print("*** MNIST ***")
print("\tNearest-Centroid: " + str(nc_mnist_score))
print("\tNearest-Centroid w/ PCA: " + str(pca_nc_mnist_score))
print("\tNearest Subclass Centroid (2): " + str(nsc_2_mnist_score))
print("\tNearest Subclass Centroid (3): " + str(nsc_3_mnist_score))
print("\tNearest Subclass Centroid (5): " + str(nsc_5_mnist_score))

print("\n*** ORL ***")
print("\tNearest-Centroid: " + str(nc_orl_score))
print("\tNearest-Centroid w/ PCA: " + str(pca_nc_orl_score))
print("\tNearest Subclass Centroid (2): " + str(nsc_2_orl_score))
print("\tNearest Subclass Centroid (3): " + str(nsc_3_orl_score))
#print("\tNearest Subclass Centroid (5): " + str(nsc_5_orl_score))