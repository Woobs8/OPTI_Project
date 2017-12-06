from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import add_dummy_feature

""" 
Nearest Centroid Algorithm
"""
class NC:
    def __init__(self):
        self.clf = NearestCentroid()

    """ 
    Calculates the mean of each class in the training data. 
    param:
        @train_data: training data
        @train_lbls: training labels
    """
    def fit(self, train_data, train_lbls):
        self.clf.fit(train_data, train_lbls)
        return self

    """ 
    Classifies test data using the class means of the training and Nearest Centroid algorithm.
    param:
        @test_data: testing data
        @test_lbls: testing labels
    returns:
        @classification: numpy array with classification labels
        @score: the mean accuracy classifications
    """
    def predict(self, test_data, test_lbls):
        classification = self.clf.predict(test_data)
        try:
            score = accuracy_score(test_lbls, classification)
        except ValueError:
            score = None

        return classification, score


""" 
Nearest Sublcass Centroid
"""
class NSC:
    """
    Initialize algorithm with hyper parameters
    param:
        @subclass_count: number of subclasses of each class
    """
    def __init__(self, subclass_count):
        self.kmeans = KMeans(n_clusters=subclass_count)
        self.subclass_centers = []
        self.label_offset = 0
        self.classes = []
        self.subclass_count = subclass_count

    """ 
    Applies K-means clustering algorithm, to cluster each class in the training data into N subclasses. 
    param:
        @train_data: training data
        @train_lbls: training labels
    """
    def fit(self, train_data, train_lbls):
        # Create set of training classes
        self.classes = list(set(train_lbls))
        class_count = len(self.classes)
        n_samples, n_features = train_data.shape

        # Iterate classes and apply K-means to find subclasses of each class
        grouped_train_data = [None] * class_count
        self.subclass_centers = np.zeros((class_count, self.subclass_count, n_features))
        self.label_offset = self.classes[0]   # Account for classifications which doesn't start at 0
        for label in self.classes:
            index = label - self.label_offset

            # Group training samples into lists for each class
            grouped_train_data[index] = [x for i, x in enumerate(train_data) if train_lbls[i]==label]

            # Apply K-means clustering algorithm to find subclasses
            self.kmeans.fit(grouped_train_data[index])
            self.subclass_centers[index] = self.kmeans.cluster_centers_
        return self

    """ 
    Classifies test samples to the classes corresponding to the nearest subclass.
    param:
        @test_data: testing data
        @test_lbls: testing labels
    returns:
        @classification: numpy array with classification labels
        @score: the mean accuracy classifications
    """
    def predict(self, test_data, test_lbls):
        class_count = len(self.classes)
        n_samples, n_features = test_data.shape
        distances = np.zeros((class_count, n_samples))

        # Iterate classes and calculate distances to each subclass centroid
        for k in range(class_count):
            class_distances = np.sqrt(((test_data - (self.subclass_centers[k,:,:])[:,np.newaxis,:]) ** 2).sum(axis=2))
            distances[k] = np.min(class_distances, axis=0)

        # Classify samples to class with closes subclass centroid
        classification = np.argmin(distances,axis=0) + self.label_offset

        # Determine classification errors by comparing classification with known labels
        try:
            score = accuracy_score(test_lbls, classification)
        except ValueError:
            score = None

        return np.asarray(classification), score


""" 
Nearest Neighbor
"""
class NN:
    """
    Initialize algorithm with hyper parameters
    param:
        @neighbor_count: number of neighbors to consider when classifying samples
        @neighbor_weight: 'uniform' / 'distance' / [callable] (weight function used for prediction)
        @n_jobs: number of parallel jobs to run (each taking up 1 cpu core)
    """
    def __init__(self, neighbor_count, neighbor_weight, n_jobs):
        self.clf = KNeighborsClassifier(neighbor_count, weights=neighbor_weight, n_jobs=n_jobs)

    """
    Prepare Nearest Neighbor algorithm
    param:
        @train_data: training data
        @train_lbls: training labels
    """
    def fit(self, train_data, train_lbls):
        self.clf.fit(train_data, train_lbls)
        return self

    """
    Classifies the test data using a Nearest Neighbor algorithm
    param:
        @test_data: testing data
        @test_lbls: testing labels
        @classification: 'hard' / 'soft' (return classified samples or classification probabilities)
    returns:
        @classification: numpy array with classification labels or classification probabilities
        @score: the mean accuracy classifications
    """
    def predict(self, test_data, test_lbls, classification="hard"):
        if classification == 'hard':
            classification = self.clf.predict(test_data)
        elif classification =='soft':
            classification = self.clf.predict_proba(test_data)

        try:
            score = accuracy_score(test_lbls, classification)
        except ValueError:
            score = None
        return classification, score


""" 
Backpropagation Perceptron
"""
class BP_Perceptron:
    def __init__(self):
        self.label_offset = 0
        self.W = np.zeros(1)

    """ 
    Trains an OVR multi-class perceptron using backpropagation.
    param:
        @train_data: training data
        @train_lbls: training labels
        @eta: learning rate
        @max_iter: maximum training iterations 
    """
    def fit(self, train_data, train_lbls, eta=1, max_iter=1000):
        # Create set of training classes
        classes = np.unique(train_lbls)
        class_count = len(classes)

        # Convert samples to float for faster numpy processing
        train_data = train_data.astype(float)

        # Augment data with bias to simplify linear discriminant function
        X = add_dummy_feature(train_data)
        n_features = len(X[0])

        # Determine discriminant hyperplane for each OVR binary classification
        self.W = np.zeros((class_count, n_features), dtype=np.float)
        label_offset = classes[0]  # Account for classifications which doesn't start at 0
        for label in classes:
            # Initialize w
            w = np.zeros(n_features, dtype=np.float)

            # Initialize OVR (One vs Rest) binary classification
            ovr_lbls = np.array([1 if lbl == label else -1 for lbl in train_lbls], dtype=np.float)

            # Batch perceptron training
            for t in range(max_iter):
                delta = 0
                chi = []
                for i, x in enumerate(X):
                    # Evaluate perceptron criterion function
                    if (np.dot(x, w) * ovr_lbls[i]) <= 0:
                        chi.append(x)
                        # Sum error terms of misclassified samples
                        delta += x * ovr_lbls[i]

                # No classification errors, algorithm is done
                if len(chi) == 0:
                    break

                # Update w
                w = w + eta * delta

            # Assign w to label-based index
            index = label - label_offset
            self.W[index] = w

        # # Create set of training classes
        # classes = np.unique(train_lbls)
        # class_count = len(classes)
        #
        # # Convert samples to float for faster numpy processing
        # train_data = train_data.astype(float)

        # # Augment data with bias to simplify linear discriminant function
        # X = add_dummy_feature(train_data).transpose()
        # n_features, n_samples = X.shape
        #
        # # Determine discriminant hyperplane for each OVR binary classification
        # self.W = np.zeros((class_count, n_features), dtype=np.float)
        #
        # # Initialize labels for OVR (One vs Rest) binary classification
        # ovr_lbls = np.where(train_lbls[np.newaxis, :] == classes[:, np.newaxis], 1, -1)
        # print(self.W.shape)
        # print(X.shape)
        # # Evaluate perceptron criterion function
        # test = np.multiply(ovr_lbls, np.dot(self.W, X))
        # print(test.shape)

        return self

    """ 
    Classifies test data using a linear discriminant function and the trained weight matrix
    param:
        @test_data: test data
        @test_lbls: test labels
    returns:
        @classification: numpy array with classification labels
        @score: the mean accuracy classifications
    """
    def predict(self, test_data, test_lbls):
        return perceptron_classify(self.W, test_data, test_lbls)


""" 
Mean-Square-Error Perceptron
"""
class MSE_Perceptron:
    def __init__(self):
        self.label_offset = 0
        self.W = np.zeros(1)

    """ 
    Optimizes an OVR multi-class perceptron by minimizing the MSE (Mean-Square-Error) using the gradient.
    param:
        @train_data: training data
        @train_lbls: training labels
        @epsilon: scaling of regularized pseudo-inverse matrix
    """
    def fit(self, train_data, train_lbls, epsilon):
        # Create set of training classes
        classes = np.unique(train_lbls)
        class_count = len(classes)

        # Convert samples to float for faster numpy processing
        train_data = train_data.astype(float)

        # Augment data with bias to simplify linear discriminant function
        X = add_dummy_feature(train_data).transpose()
        n_features, n_samples = X.shape

        # Calculate regularized pseudo-inverse of X
        X_pinv = np.dot(np.linalg.inv(np.dot(X, X.transpose()) + epsilon * np.identity(n_features)), X)

        # Determine discriminant hyperplane for each OVR binary classification
        self.W = np.zeros((class_count, n_features), dtype=np.float)

        # Initialize target matrix B for OVR (One vs Rest) binary classification
        B = np.where(train_lbls[np.newaxis, :] == classes[:, np.newaxis], 1, -1)

        # Calculate optimized weight vectors
        self.W = np.dot(B, X_pinv.transpose())
        return self

    """ 
    Classifies test data using a linear discriminant function and the trained weight matrix
    param:
        @test_data: test data
        @test_lbls: test labels
    returns:
        @classification: numpy array with classification labels
        @score: the mean accuracy classifications
    """
    def predict(self, test_data, test_lbls):
        return perceptron_classify(self.W, test_data, test_lbls)


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
    classes = np.unique(test_lbls)
    label_offset = classes[0]  # Account for classifications which doesn't start at 0

    # Convert samples to float for faster numpy processing
    test_data = test_data.astype(float)

    # Augment data with bias to simplify linear discriminant function
    X = add_dummy_feature(test_data).transpose()

    decision = np.dot(W,X)
    classification = np.argmax(decision,axis=0)+label_offset

    try:
        score = accuracy_score(test_lbls, classification)
    except ValueError:
        score = None
    return classification, score