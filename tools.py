from sklearn.decomposition import PCA
from numpy import reshape

"""" 
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

    pca_train_data = []
    # Iterate samples due to memory constraint
    for sample in train_data:
        # Optimize model to data and transform data
        pca_train_data.append(pca.fit_transform(sample.reshape(1,-1)))

    pca_test_data = []
    # Iterate samples due to memory constraint
    for sample in test_data:
        # Optimize model to data and transform data
        pca_test_data.append(pca.fit_transform(sample.reshape(1,-1)))

    return pca_train_data, pca_test_data

def visualize_data(data):
    print("not implemented yet")