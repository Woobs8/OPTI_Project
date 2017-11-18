import os
import sys
from tools import loadMNIST, loadORL, pca, plot_mnist_centroids, plot_orl_centroids, plot_2D_data, subplot_2D_data
from classify import nc, nsc, nn
import matplotlib.pyplot as plt


def main(run_mnist = True, run_orl = True, run_nc = True, run_nsc = True, run_nn = True):

    """ ********* Loading ORL samples ********* """
    if run_mnist:
        # MNIST sample directory
        MNIST_DIR = "MNIST"
        MNIST_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + MNIST_DIR
        # Load MNIST samples
        mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls = loadMNIST(MNIST_PATH)
        # Apply PCA to MNIST samples
        pca_mnist_train_images, pca_mnist_test_images = pca(mnist_train_images, mnist_test_images)

    """ ********* Loading ORL samples ********* """
    if run_orl:
        # ORL sample directory
        ORL_DIR = "ORL"
        ORL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + ORL_DIR
        # Load ORL samples
        orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls = loadORL(ORL_PATH)
        # Apply PCA to ORL samples
        pca_orl_train_images, pca_orl_test_images = pca(orl_train_images, orl_test_images)


    """ ********* Classifying MNIST samples ********* """
    if run_mnist:
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
            # 2 subclasses, PCA data
            pca_nsc_2_mnist_class, pca_nsc_2_mnist_score = nsc(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images, mnist_test_lbls,2)
            # 3 subclasses, 784D data
            nsc_3_mnist_class, nsc_3_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,3)
            # 3 subclasses, PCA data
            pca_nsc_3_mnist_class, pca_nsc_3_mnist_score = nsc(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images, mnist_test_lbls,3)
            # 5 subclasses, 784D data
            nsc_5_mnist_class, nsc_5_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,5)
            # 5 subclasses, PCA data
            pca_nsc_5_mnist_class, pca_nsc_5_mnist_score = nsc(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images, mnist_test_lbls,5)

        # Nearest Neighbor
        if run_nn:
            nn_mnist_class, nn_mnist_prob, nn_mnist_score = nn(mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls,1,'uniform')
            pca_nn_mnist_class, pca_nn_mnist_prob, pca_nn_mnist_score = nn(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images, mnist_test_lbls,1,'uniform')


    """ ********* Classifying ORL samples ********* """
    if run_orl:
        # Nearest Centroid
        if run_nc:
            # 784D data
            nc_orl_class, nc_orl_score = nc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls)
            # PCA data
            pca_nc_orl_class, pca_nc_orl_score = nc(pca_orl_train_images, orl_train_lbls.ravel(), pca_orl_test_images, orl_test_lbls)

        # Nearest Subclass Centroid
        if run_nsc:
            # 2 subclasses, 1200D data
            nsc_2_orl_class, nsc_2_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,2)
            # 2 subclasses, PCA data
            pca_nsc_2_orl_class, pca_nsc_2_orl_score = nsc(pca_orl_train_images, orl_train_lbls.ravel(), pca_orl_test_images, orl_test_lbls,2)
            # 3 subclasses, 1200D data
            nsc_3_orl_class, nsc_3_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,3)
            # 3 subclasses, PCA data
            pca_nsc_3_orl_class, pca_nsc_3_orl_score = nsc(pca_orl_train_images, orl_train_lbls.ravel(), pca_orl_test_images, orl_test_lbls,3)
            # 5 subclasses, 1200D data
            #nsc_5_orl_class, nsc_5_orl_score = nsc(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,5)
            # 5 subclasses, PCA data
            #pca_nsc_5_orl_class, pca_nsc_5_orl_score = nsc(pca_orl_train_images, orl_train_lbls.ravel(), pca_orl_test_images, orl_test_lbls,5)

        # Nearest Neighbor
        if run_nn:
            nn_orl_class, nn_orl_prob, nn_orl_score = nn(orl_train_images, orl_train_lbls.ravel(), orl_test_images, orl_test_lbls,1,'uniform')
            pca_nn_orl_class, pca_nn_orl_prob, pca_nn_orl_score = nn(pca_orl_train_images, orl_train_lbls.ravel(), pca_orl_test_images, orl_test_lbls,1,'uniform')


    """ ********* Data Visualization ********* """
    if show_figs:
        """ ********* MNIST ********* """
        if run_mnist:
            # Training data
            plot_mnist_centroids(mnist_train_images, mnist_train_lbls, 'MNIST Training Data Centroids')

            # Test data
            plot_mnist_centroids(mnist_test_images, mnist_test_lbls, 'MNIST Test Data Centroids')

            # PCA Training data
            plot_2D_data(pca_mnist_train_images, mnist_train_lbls, 'MNIST PCA Training Data')

            # Classified PCA test data
            if run_nc: plot_2D_data(pca_mnist_test_images, pca_nc_mnist_class,
                                           'NC Classified MNIST PCA Test Data')
            if run_nsc: subplot_2D_data(pca_mnist_test_images,
                                          [pca_nsc_2_mnist_class, pca_nsc_3_mnist_class, pca_nsc_5_mnist_class],
                                          'NSC Classified MNIST PCA Test Data',
                                          ['2 Subclasses', '3 Subclasses', '5 Subclasses'])

            if run_nn: plot_2D_data(pca_mnist_test_images, pca_nc_mnist_class,
                                       'NN Classified MNIST PCA Test Data')

            # Labeled (actual) PCA test data
            plot_2D_data(pca_mnist_test_images, mnist_test_lbls, 'Labeled MNIST PCA Test Data')

        """ ********* ORL ********* """
        if run_orl:
            # Training data
            plot_orl_centroids(orl_train_images, orl_train_lbls, 'ORL Training Data Centroids')

            # Test data
            plot_orl_centroids(orl_test_images, orl_test_lbls.ravel(), 'ORL Test Data Centroids')

            # PCA Training data
            plot_2D_data(pca_orl_train_images, orl_train_lbls.ravel(), 'ORL PCA Training Data')

            # Classified PCA test data
            if run_nc: plot_2D_data(pca_orl_test_images, pca_nc_orl_class, 'NC Classified ORL PCA Test Data')

            if run_nsc: subplot_2D_data(pca_orl_test_images, [pca_nsc_2_orl_class, pca_nsc_3_orl_class],
                                'NSC Classified ORL PCA Test Data', ['2 Subclasses', '3 Subclasses'])

            if run_nn: plot_2D_data(pca_orl_test_images, pca_nc_orl_class, 'NN Classified ORL PCA Test Data')

            # Labeled (actual) PCA test data
            plot_2D_data(pca_orl_test_images, orl_test_lbls.ravel(), 'Labeled ORL PCA Test Data')


    """ ********* Classification scores ********* """
    print("*** Classification Scores ***")
    if run_mnist:
        print("*** MNIST ***")
        # Nearest Centroid
        if run_nc:
            print("\tNearest-Centroid: " + str(nc_mnist_score))
            print("\tNearest-Centroid w/ PCA: " + str(pca_nc_mnist_score))

        # Nearest Subclass Centroid
        if run_nsc:
            print("\tNearest Subclass Centroid (2): " + str(nsc_2_mnist_score))
            print("\tNearest Subclass Centroid (2) w/ PCA: " + str(pca_nsc_2_mnist_score))
            print("\tNearest Subclass Centroid (3): " + str(nsc_3_mnist_score))
            print("\tNearest Subclass Centroid (3) w/ PCA: " + str(pca_nsc_3_mnist_score))
            print("\tNearest Subclass Centroid (5): " + str(nsc_5_mnist_score))
            print("\tNearest Subclass Centroid (5) w/ PCA: " + str(pca_nsc_5_mnist_score))

        # Nearest Neighbor
        if run_nn:
            print("\tNearest Neighbor: " + str(nn_mnist_score))
            print("\tNearest Neighbor w/ PCA: " + str(pca_nn_mnist_score))

    if run_orl:
        print("\n*** ORL ***")
        # Nearest Centroid
        if run_nc:
            print("\tNearest-Centroid: " + str(nc_orl_score))
            print("\tNearest-Centroid w/ PCA: " + str(pca_nc_orl_score))

        # Nearest Subclass Centroid
        if run_nsc:
            print("\tNearest Subclass Centroid (2): " + str(nsc_2_orl_score))
            print("\tNearest Subclass Centroid (2) w/ PCA: " + str(pca_nsc_2_orl_score))
            print("\tNearest Subclass Centroid (3): " + str(nsc_3_orl_score))
            print("\tNearest Subclass Centroid (3) w/ PCA: " + str(pca_nsc_3_orl_score))
            #print("\tNearest Subclass Centroid (5): " + str(nsc_5_orl_score))

        # Nearest Neighbor
        if run_nn:
            print("\tNearest Neighbor: " + str(nn_orl_score))
            print("\tNearest Neighbor w/ PCA: " + str(pca_nn_orl_score))

    # Flush results to stdout
    sys.stdout.flush()

    # Block scriot to keep figures
    plt.show()

if __name__ == "__main__":
    # set which algorithm to apply for this execution
    run_nc = False
    run_nsc = False
    run_nn = False
    show_figs = True
    run_mnist = False
    run_orl = False
    if len(sys.argv[1]) > 1:
        for arg in sys.argv:
            if arg == 'nc' or arg == 'NC':
                run_nc = True
            elif arg == 'nsc' or arg == 'NSC':
                run_nsc = True
            elif arg == 'nn' or arg == 'NN':
                run_nn = True
            elif arg == 'no-fig':
                show_figs = False
            elif arg == 'mnist' or arg == 'MNIST':
                run_mnist = True
            elif arg == 'orl' or arg == 'ORL':
                run_orl = True
    else:
        run_nc = True
        run_nsc = True
        run_nn = True
        run_mnist = True
        run_orl = True

    if (not run_mnist) and (not run_orl):
        run_mnist = True
        run_orl = True

    main(run_mnist, run_orl, run_nc, run_nsc, run_nn)