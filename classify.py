from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import add_dummy_feature

""" 
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
    clf = NearestCentroid()
    clf.fit(train_data, train_lbls)
    classification = clf.predict(test_data)
    try:
        score = accuracy_score(test_lbls, classification)
    except ValueError:
        score = 0

    return classification, score


""" 
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
    score = accuracy_score(test_lbls, classification)

    return np.asarray(classification), score


"""
Classifies the test data using a Nearest Neighbor algorithm
param:
    @train_data: training data
    @train_lbls: training labels
    @test_data: testing data
    @test_lbls: testing labels
    @neighbor_count: number of neighbors to consider when classifying samples
    @neighbor_weight: 'uniform' / 'distance' / [callable] (weight function used for prediction)
    @n_jobs: number of parallel jobs to run (each taking up 1 cpu core)
    @classification: 'hard' / 'soft' (return classified samples or classification probabilities)
returns:
    @classification: numpy array with classification labels or classification probabilities
    @score: the mean accuracy classifications
"""
def nn(train_data, train_lbls, test_data, test_lbls, neighbor_count, neighbor_weight='uniform', n_jobs=1, classification="hard"):
    clf = KNeighborsClassifier(neighbor_count, weights=neighbor_weight, n_jobs=n_jobs)
    clf.fit(train_data, train_lbls)
    if classification == 'hard':
        classification = clf.predict(test_data)
    elif classification =='soft':
        classification = clf.predict_proba(test_data)
    score = accuracy_score(test_lbls, classification)
    return classification, score


""" 
Trains a benchmark perceptron using the Scikit-learn implemented training data, and classifies the test data using the 
trained perceptron model. 
param:
    @train_data: training data
    @train_lbls: training labels
    @test_data: testing data
    @test_lbls: testing labels
    @eta: learning rate
    @n_jobs: number of parallel jobs to run (each taking up 1 cpu core)
    @max_iter: maximum training iterations 
returns:
    @classification: numpy array with classification labels or classification probabilities
    @score: the mean accuracy classifications
"""
def perceptron_benchmark(train_data, train_lbls, test_data, test_lbls, eta=1, n_jobs=1, max_iter=1000):
    clf = Perceptron(eta0=eta, n_jobs=n_jobs, shuffle=True, max_iter=max_iter)
    clf.fit(train_data, train_lbls)
    classification = clf.predict(test_data)

    score = accuracy_score(test_lbls, classification)
    return classification, score


""" 
Trains an OVR multi-class perceptron using backpropagation.
param:
    @train_data: training data
    @train_lbls: training labels
    @eta: learning rate
    @max_iter: maximum training iterations 
returns:
    @W: trained OVR weight matrix
"""
def perceptron_bp(train_data, train_lbls, eta=1, max_iter=1000):
    # Create set of training classes
    classes = list(set(train_lbls))
    class_count = len(classes)

    # Convert samples to float for faster numpy processing
    train_data = train_data.astype(float)

    # Augment data with bias to simplify linear discriminant function
    aug_train_data = add_dummy_feature(train_data)
    aug_feature_count = len(aug_train_data[0])

    # Determine discriminant hyperplane for each OVR binary classification
    W = np.zeros((class_count,aug_feature_count), dtype=np.float)
    label_offset = classes[0]   # Account for classifications which doesn't start at 0
    for label in classes:
        # Initialize w
        w = np.zeros(aug_feature_count, dtype=np.float)

        # Initialize OVR (One vs Rest) binary classification
        ovr_lbls = np.array([1 if lbl == label else -1 for lbl in train_lbls], dtype=np.float)

        # Batch perceptron training
        for t in range(max_iter):
            delta = 0
            for i,x in enumerate(aug_train_data):
                # Evaluate perceptron criterion function
                if (np.dot(x,w)*ovr_lbls[i]) <= 0:
                    # Sum error terms of misclassified samples
                    delta += x*ovr_lbls[i]

            # No classification error, algorithm is done
            if not np.count_nonzero(delta):
                break

            # Update w
            w = w + eta*delta

        # Assign w to label-based index
        index = label - label_offset
        W[index] = w

    return W


""" 
Optimizes an OVR multi-class perceptron by minimizing the MSE (Mean-Square-Error) using the gradient.
param:
    @train_data: training data
    @train_lbls: training labels
    @epsilon: scaling of regularized pseudo-inverse matrix
returns:
    @W: trained OVR weight matrix
"""
def perceptron_mse(train_data, train_lbls, epsilon):
    # Create set of training classes
    classes = list(set(train_lbls))
    class_count = len(classes)

    # Convert samples to float for faster numpy processing
    train_data = train_data.astype(float)

    # Augment data with bias to simplify linear discriminant function
    aug_train_data = add_dummy_feature(train_data)
    aug_feature_count = len(aug_train_data[0])
    X = aug_train_data.transpose()

    # Calculate regularized pseudo-inverse of X
    X_reg = np.dot(np.linalg.inv(np.dot(X, X.transpose()) + epsilon * np.identity(aug_feature_count)), X)

    # Determine discriminant hyperplane for each OVR binary classification
    W = np.zeros((class_count,aug_feature_count), dtype=np.float)
    label_offset = classes[0]   # Account for classifications which doesn't start at 0
    for label in classes:
        # Initialize OVR (One vs Rest) binary classification
        b = np.array([1 if lbl == label else -1 for lbl in train_lbls], dtype=np.float)

        # Calculate optimized weight vector
        w = np.dot(X_reg, b)

        # Assign w to label-based index
        index = label - label_offset
        W[index] = w
    return W


""" 
Applies OVR to classify test data using a trained weight matrix.
param:
    @W: weight matrix
    @test_data: test data
    @test_lbls: test labels
returns:
    @classification: numpy array with classification labels
    @score: the mean accuracy classifications
"""
def perceptron_classify(W, test_data, test_lbls):
    # Create set of training classes
    classes = list(set(test_lbls))
    label_offset = classes[0]  # Account for classifications which doesn't start at 0

    # Convert samples to float for faster numpy processing
    test_data = test_data.astype(float)

    # Augment data with bias to simplify linear discriminant function
    aug_test_data = add_dummy_feature(test_data)

    decision = np.dot(W,aug_test_data.transpose())
    classification = np.argmax(decision,axis=0)+label_offset

    score = accuracy_score(test_lbls, classification)
    return classification, score