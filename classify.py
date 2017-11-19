from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
from tools import flatten_array

"""" 
Calculates the mean of each class in the training data. 
Test samples are then classified using a Nearest Centroid algorithm.
param:
    @train_data: training data
    @train_lbls: training labels
    @test_data: testing data
    @test_lbls: testing labels
returns:
    @classification: numpy array with classification labels
    @score: the mean accuracy classifications
"""
def nc(train_data, train_lbls, test_data, test_lbls):
    train_lbls = flatten_array(train_lbls)
    test_lbls = flatten_array(test_lbls)

    clf = NearestCentroid()
    clf.fit(train_data, train_lbls)
    classification = clf.predict(test_data)
    score = clf.score(test_data,test_lbls)
    return classification, score


"""" 
Applies K-means clustering algorithm, to cluster each class in the training data into N subclasses. 
Test samples are then classified to the class corresponding to the nearest subclass.
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
    train_lbls = flatten_array(train_lbls)
    test_lbls = flatten_array(test_lbls)

    # Create set of training classes
    classes = list(set(train_lbls))
    class_count = len(classes)

    # Iterate classes and apply K-means to find subclasses of each class
    kmeans = KMeans(n_clusters=subclass_count)
    grouped_train_data = [None] * class_count
    subclass_centers = [None] * class_count
    label_offset = classes[0]   # Account for classifications which doesn't start at 0
    for label in classes:
        index = label - label_offset

        # Group training samples into lists for each class
        grouped_train_data[index] = [x for i, x in enumerate(train_data) if train_lbls[i]==label]

        # Apply K-means clustering algorithm to find subclasses
        kmeans.fit(grouped_train_data[index])
        subclass_centers[index] = kmeans.cluster_centers_

    # Iterate samples and calculate distance to subclass cluster centers
    test_sample_count = len(test_data)
    classification = [None]*test_sample_count
    for i, sample in enumerate(test_data):
        min = None
        # Iterate base classes
        for j, class_centers in enumerate(subclass_centers):
            label = j + label_offset
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
    classification_errors = [x for i, x in enumerate(classification) if classification[i] != test_lbls[i]]
    class_err_count = len(classification_errors)
    score = 1-class_err_count / test_sample_count

    return np.asarray(classification), score


"""" 
Fit model to training data and classify the test data using a Nearest Neighbor algorithm
param:
    @train_data: training data
    @train_lbls: training labels
    @test_data: testing data
    @test_lbls: testing labels
    @neighbor_weight: weight function used for prediction
    @n_jobs: number of parallel jobs to run (each taking up 1 cpu core)
returns:
    @classification: numpy array with classification labels
    @probabilities: numpy array with probability estimates for the test data
    @score: the mean accuracy classifications
"""
def nn(train_data, train_lbls, test_data, test_lbls, neighbor_count, neighbor_weight='uniform', n_jobs=1):
    train_lbls = flatten_array(train_lbls)
    test_lbls = flatten_array(test_lbls)

    clf = KNeighborsClassifier(neighbor_count, weights=neighbor_weight, n_jobs=n_jobs)
    clf.fit(train_data, train_lbls)
    classification = clf.predict(test_data)
    probabilities = clf.predict_proba(test_data)
    score = clf.score(test_data, test_lbls)
    return classification, probabilities, score