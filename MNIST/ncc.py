import matplotlib.pyplot as plt
from sklearn import neighbors
from mnist import MNIST
from numpy import reshape
mndata = MNIST('samples')
images, labels = mndata.load_training()

reshaped_images = []
for image in images:
    reshaped_images.append(reshape(image,[28,28]))

clf = neighbors.NearestCentroid()
param = clf.fit(reshaped_images,labels)
clf_samples = clf.predict(reshaped_images)
#clf.score(images,labels)
