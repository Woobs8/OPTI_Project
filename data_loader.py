from mnist import MNIST
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split

"""" 
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


"""" 
Load ORL dataset from directory and splits it training and testing datasets
param:
    @fp: path to directory containing data
    @test_size: size of test dataset (0-1)
returns:
    train_data, train_lbls, test_data, test_lbls
"""
def loadORL(fp, test_size=0.3):
    data = np.array(scio.loadmat(fp+'/orl_data.mat')['data']).transpose()
    lbls = np.array(scio.loadmat(fp+'/orl_lbls.mat')['lbls'])

    # Split data into training and testing datasets
    train_data, test_data, train_lbls, test_lbls = train_test_split(data, lbls, test_size=test_size)

    return train_data, train_lbls, test_data, test_lbls

