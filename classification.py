from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans
import numpy as np

"""" 
Fit model to training data and classify the test data using a Nearest Centroid algorithm
param:
    @train_data: training data
    @train_lbls: training labels
    @test_data: testing data
    @test_lbls: testing labels
returns:
    @classification: numpy array with classification labels
    @score: the mean accuracy classifications
"""
def nc(train_data, train_label, test_data, test_lbls):
    clf = NearestCentroid()
    clf.fit(train_data, train_label.ravel())
    classification = clf.predict(test_data)
    score = clf.score(test_data,test_lbls)
    return classification, score


"""" 
Applies a K-means clustering algorithm on the training data to identity subclasses of each class, and then classifies 
the test data using a Nearest Subclass Centroid algorithm
param:
    @train_data: training data
    @train_lbls: training labels
    @test_data: testing data
    @test_lbls: testing labels
    @subclass_count: number of subclasses of each class
returns:
    @classification: numpy array with classification labels
    @score: the mean accuracy classifications
"""
def nsc(train_data, train_lbls, test_data, test_lbls, subclass_count):
    # Create set of training classes
    classes = list(set(train_lbls))
    class_count = len(classes)

    # Iterate classes and apply K-means to find subclasses of each class
    kmeans = KMeans(n_clusters=subclass_count)
    grouped_train_data = [None] * class_count
    subclass_centers = [None] * class_count * subclass_count
    for label in classes:
        # Group training samples into lists for each class
        grouped_train_data[label-1] = [x for i, x in enumerate(train_data) if train_lbls[i]==label]

        # Apply K-means clustering algorithm to find subclasses
        kmeans.fit(grouped_train_data[label-1])
        subclass_centers[label-1] = kmeans.cluster_centers_

    # Iterate samples and calculate distance to subclass cluster centers
    test_sample_count = len(test_data)
    classification = [None]*test_sample_count
    for i, sample in enumerate(test_data):
        min = None
        # Iterate classes
        for j, class_centers in enumerate(subclass_centers):
            label = j+1 # classes start at 1, index starts at 0
            if class_centers is not None:
                # Iterate centroids of subclasses
                for subclass_center in class_centers:
                    # Calculate distance to centroid
                    dist = np.linalg.norm(subclass_center-sample)

                    # Classify sample as class corresponding to subclass if distance is lowest encountered
                    if((min is None) or (dist < min)):
                        min = dist
                        classification[i]=label

    # Determine classification errors by comparing classification with known labels
    classification_errors = [x for i, x in enumerate(classification) if classification[i] == test_lbls[i]]
    class_err_count = len(classification_errors)
    score = class_err_count / test_sample_count

    return classification, score