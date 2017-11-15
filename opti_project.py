import os
import sys
from tools import loadMNIST, loadORL, pca
from classification import nc, nsc


def main(run_nc = True, run_nsc = True, run_nn = True):
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
    if run_nc:
        # 784D data
        nc_mnist_class, nc_mnist_score = nc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls)
        # PCA data
        pca_nc_mnist_class, pca_nc_mnist_score = nc(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images, mnist_test_lbls)

    # Nearest Subclass Centroid
    if run_nsc:
        # 2 subclasses, 784D data
        nsc_2_mnist_class, nsc_2_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,2)
        # 3 subclasses, 784D data
        nsc_3_mnist_class, nsc_3_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,3)
        # 5 subclasses, 784D data
        nsc_5_mnist_class, nsc_5_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,5)


    """ ********* Classifying ORL samples ********* """
    # Nearest Centroid
    if run_nc:
        nc_orl_class, nc_orl_score = nc(orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls)
        # Nearest Centroid w/ PCA data
        pca_nc_orl_class, pca_nc_orl_score = nc(pca_orl_train_images, orl_train_lbls, pca_orl_test_images, orl_test_lbls)

    # Nearest Subclass Centroid
    if run_nsc:
        # 2 subclasses, 1200D data
        nsc_2_orl_class, nsc_2_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,2)
        # 3 subclasses, 1200D data
        nsc_3_orl_class, nsc_3_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,3)
        # 5 subclasses, 1200D data
        #nsc_5_orl_class, nsc_5_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,5)


    """ ********* Classification scores ********* """
    print("*** Classification Scores ***")
    print("*** MNIST ***")
    # Nearest Centroid
    if run_nc:
        print("\tNearest-Centroid: " + str(nc_mnist_score))
        print("\tNearest-Centroid w/ PCA: " + str(pca_nc_mnist_score))

    # Nearest Subclass Centroid
    if run_nsc:
        print("\tNearest Subclass Centroid (2): " + str(nsc_2_mnist_score))
        print("\tNearest Subclass Centroid (3): " + str(nsc_3_mnist_score))
        print("\tNearest Subclass Centroid (5): " + str(nsc_5_mnist_score))

    print("\n*** ORL ***")
    # Nearest Centroid
    if run_nc:
        print("\tNearest-Centroid: " + str(nc_orl_score))
        print("\tNearest-Centroid w/ PCA: " + str(pca_nc_orl_score))

    # Nearest Subclass Centroid
    if run_nsc:
        print("\tNearest Subclass Centroid (2): " + str(nsc_2_orl_score))
        print("\tNearest Subclass Centroid (3): " + str(nsc_3_orl_score))
        #print("\tNearest Subclass Centroid (5): " + str(nsc_5_orl_score))


if __name__ == "__main__":
    # set which algorithm to apply for this execution
    run_nc = False
    run_nsc = False
    run_nn = False
    if len(sys.argv[1]) > 1:
        for arg in sys.argv:
            if arg == 'nc':
                run_nc = True
            elif arg == 'nsc':
                run_nsc = True
            elif arg == 'nn':
                run_nn = True
    else:
        run_nc = True
        run_nsc = True
        run_nn = True

    main(run_nc, run_nsc, run_nn)