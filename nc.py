from sklearn.neighbors import NearestCentroid

def classify(train_data, train_label, test_data, test_lbls):
    clf = NearestCentroid()
    clf.fit(train_data, train_label)
    classification = clf.predict(test_data)
    score = clf.score(test_data,test_lbls)
    return classification, score
