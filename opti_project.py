import matplotlib.pyplot as plt
import data_loader
import os
import nc
from tools import pca

""" ********* Loading MNIST samples ********* """
# MNIST sample directory
MNIST_DIR = "MNIST"

MNIST_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + MNIST_DIR
mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls = data_loader.loadMNIST(MNIST_PATH)
#pca_mnist_train_images, pca_mnist_test_images = pca(mnist_train_images, mnist_test_images)


""" ********* Loading ORL samples ********* """
# ORL sample directory
ORL_DIR = "ORL"
ORL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + ORL_DIR
orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls = data_loader.loadORL(ORL_PATH)
#pca_orl_train_images, pca_orl_test_images = pca(orl_train_images, orl_test_images)


""" ********* Classifying MNIST samples ********* """
# Nearest Centroid
nc_mnist_class, nc_mnist_score = nc.classify(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls)
# Nearest Centroid w/ PCA data
#pca_nc_mnist_class, pca_nc_mnist_score = nc.classify(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images, mnist_test_lbls)

""" ********* Classifying ORL samples ********* """
# Nearest Centroid
nc_orl_class, nc_orl_score = nc.classify(orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls)


""" ********* Classification scores ********* """
print("*** Classification Scores ***")
print("*** MNIST ***")
print("\tNearest-Centroid: " + str(nc_mnist_score))
#print("\tNearest-Centroid w/ PCA: " + str(pca_nc_mnist_score))

print("\n*** ORL ***")
print("\tNearest-Centroid: " + str(nc_orl_score))
