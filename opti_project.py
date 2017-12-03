import os
import sys
from tools import loadMNIST, loadORL, pca, tsne, plot_mnist_centroids, plot_orl_centroids, plot_2D_data, subplot_2D_data
from classify import nc, nsc, nn, perceptron_bp, perceptron_classify, perceptron_mse
import matplotlib.pyplot as plt
import multiprocessing
from os.path import exists
from os import makedirs


def main(run_mnist=True, run_orl=True, run_nc=True, run_nsc=True, run_nn=True, run_perc_bp=True, run_perc_mse=True, cpus=1):

    """ ********* Loading MNIST samples ********* """
    if run_mnist:
        # MNIST sample directory
        MNIST_DIR = "MNIST"
        MNIST_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + MNIST_DIR
        # Load MNIST samples
        mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls = loadMNIST(MNIST_PATH)
        # Apply PCA to MNIST samples
        pca_mnist_train_images, pca_mnist_test_images = pca(mnist_train_images, mnist_test_images)
        # Apply TSNE to MNIST samples
        #tsne_mnist_train_images, tsne_mnist_test_images = tsne(mnist_train_images, mnist_test_images)

    """ ********* Loading ORL samples ********* """
    if run_orl:
        # ORL sample directory
        ORL_DIR = "ORL"
        ORL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/' + ORL_DIR
        # Load ORL samples
        orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls = loadORL(ORL_PATH)
        # Apply PCA to ORL samples
        pca_orl_train_images, pca_orl_test_images = pca(orl_train_images, orl_test_images)
        # Apply TSNE to ORL samples
        #tsne_orl_train_images, tsne_orl_test_images = tsne(orl_train_images, orl_test_images)


    """ ********* Performance Parameters ********* """
    # Set parameters for parallel execution
    avail_cpus = multiprocessing.cpu_count()
    print("Utilizing " + str(cpus) + "/" + str(avail_cpus) + " CPU cores for parallel execution.")


    """ ********* Classifying MNIST samples ********* """
    if run_mnist:
        # Nearest Centroid
        if run_nc:
            # 784D data
            nc_mnist_class, nc_mnist_score = nc(mnist_train_images, mnist_train_lbls, mnist_test_images,
                                                mnist_test_lbls)
            # PCA data
            pca_nc_mnist_class, pca_nc_mnist_score = nc(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images,
                                                        mnist_test_lbls)

        # Nearest Subclass Centroid
        if run_nsc:
            # 2 subclasses, 784D data
            nsc_2_mnist_class, nsc_2_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images,
                                                       mnist_test_lbls, 2)
            # 2 subclasses, PCA data
            pca_nsc_2_mnist_class, pca_nsc_2_mnist_score = nsc(pca_mnist_train_images, mnist_train_lbls,
                                                               pca_mnist_test_images, mnist_test_lbls, 2)
            # 3 subclasses, 784D data
            nsc_3_mnist_class, nsc_3_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images,
                                                       mnist_test_lbls, 3)
            # 3 subclasses, PCA data
            pca_nsc_3_mnist_class, pca_nsc_3_mnist_score = nsc(pca_mnist_train_images, mnist_train_lbls,
                                                               pca_mnist_test_images, mnist_test_lbls, 3)
            # 5 subclasses, 784D data
            nsc_5_mnist_class, nsc_5_mnist_score = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images,
                                                       mnist_test_lbls, 5)
            # 5 subclasses, PCA data
            pca_nsc_5_mnist_class, pca_nsc_5_mnist_score = nsc(pca_mnist_train_images, mnist_train_lbls,
                                                           pca_mnist_test_images, mnist_test_lbls, 5)

        # Nearest Neighbor
        if run_nn:
            nn_mnist_class, nn_mnist_score = nn(mnist_train_images, mnist_train_lbls, mnist_test_images,
                                                               mnist_test_lbls, 1, 'uniform',cpus,'hard')
            pca_nn_mnist_class, pca_nn_mnist_score = nn(pca_mnist_train_images, mnist_train_lbls,
                                                                           pca_mnist_test_images, mnist_test_lbls, 1,
                                                                           'uniform',cpus,'hard')

        # Backpropagation Perceptron
        if run_perc_bp:
            # 784D data
            W = perceptron_bp(mnist_train_images, mnist_train_lbls, eta=0.01, max_iter=100)
            perc_bp_mnist_class, perc_bp_mnist_score = perceptron_classify(W, mnist_test_images,mnist_test_lbls)

            # PCA data
            W = perceptron_bp(pca_mnist_train_images, mnist_train_lbls, eta=0.01, max_iter=100)
            pca_perc_bp_mnist_class, pca_perc_bp_mnist_score = perceptron_classify(W, pca_mnist_test_images,mnist_test_lbls)


        # MSE Perceptron
        if run_perc_mse:
            # 1200D data
            W = perceptron_mse(mnist_train_images, mnist_train_lbls, epsilon=10 ** -6)
            perc_mse_mnist_class, perc_mse_mnist_score = perceptron_classify(W, mnist_test_images, mnist_test_lbls)

            # PCA data
            W = perceptron_mse(pca_mnist_train_images, mnist_train_lbls, epsilon=10 ** -6)
            pca_perc_mse_mnist_class, pca_perc_mse_mnist_score = perceptron_classify(W, pca_mnist_test_images,
                                                                                     mnist_test_lbls)



    """ ********* Classifying ORL samples ********* """
    if run_orl:
        # Nearest Centroid
        if run_nc:
            # 784D data
            nc_orl_class, nc_orl_score = nc(orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls)
            # PCA data
            pca_nc_orl_class, pca_nc_orl_score = nc(pca_orl_train_images, orl_train_lbls, pca_orl_test_images,
                                                    orl_test_lbls)

        # Nearest Subclass Centroid
        if run_nsc:
            # 2 subclasses, 1200D data
            nsc_2_orl_class, nsc_2_orl_score = nsc(orl_train_images, orl_train_lbls, orl_test_images,
                                                   orl_test_lbls, 2)
            # 2 subclasses, PCA data
            pca_nsc_2_orl_class, pca_nsc_2_orl_score = nsc(pca_orl_train_images, orl_train_lbls,
                                                           pca_orl_test_images, orl_test_lbls, 2)
            # 3 subclasses, 1200D data
            nsc_3_orl_class, nsc_3_orl_score = nsc(orl_train_images, orl_train_lbls, orl_test_images,
                                                   orl_test_lbls, 3)
            # 3 subclasses, PCA data
            pca_nsc_3_orl_class, pca_nsc_3_orl_score = nsc(pca_orl_train_images, orl_train_lbls,
                                                           pca_orl_test_images, orl_test_lbls, 3)
            # 5 subclasses, 1200D data
            nsc_5_orl_class, nsc_5_orl_score = nsc(orl_train_images, orl_train_lbls,
                                                           orl_test_images, orl_test_lbls, 5)
            # 5 subclasses, PCA data
            pca_nsc_5_orl_class, pca_nsc_5_orl_score = nsc(pca_orl_train_images, orl_train_lbls,
                                                           pca_orl_test_images, orl_test_lbls, 5)

        # Nearest Neighbor
        if run_nn:
            nn_orl_class, nn_orl_prob, nn_orl_score = nn(orl_train_images, orl_train_lbls, orl_test_images,
                                                         orl_test_lbls, 1, 'uniform',cpus,'hard')
            pca_nn_orl_class, pca_nn_orl_prob, pca_nn_orl_score = nn(pca_orl_train_images, orl_train_lbls,
                                                                     pca_orl_test_images, orl_test_lbls, 1, 'uniform',
                                                                     cpus,'hard')

        # Backpropagation Perceptron
        if run_perc_bp:
            # 1200D data
            W = perceptron_bp(orl_train_images, orl_train_lbls,eta=0.01, max_iter=100)
            perc_bp_orl_class, perc_bp_orl_score = perceptron_classify(W, orl_test_images, orl_test_lbls)

            # PCA data
            W = perceptron_bp(pca_orl_train_images, orl_train_lbls,eta=0.01, max_iter=100)
            pca_perc_bp_orl_class, pca_perc_bp_orl_score = perceptron_classify(W, pca_orl_test_images, orl_test_lbls)


        # MSE Perceptron
        if run_perc_mse:
            # 1200D data
            W = perceptron_mse(orl_train_images, orl_train_lbls, epsilon=10 ** -4)
            perc_mse_orl_class, perc_mse_orl_score = perceptron_classify(W, orl_test_images, orl_test_lbls)

            # PCA data
            W = perceptron_mse(pca_orl_train_images, orl_train_lbls, epsilon=10 ** -3)
            pca_perc_mse_orl_class, pca_perc_mse_orl_score = perceptron_classify(W, pca_orl_test_images, orl_test_lbls)


    """ ********* Data Visualization ********* """
    if show_figs:
        dir = 'figures/'
        if not exists(dir):
            makedirs(dir)

        """ ********* MNIST ********* """
        if run_mnist:
            mnist_dir = dir + 'mnist/'
            if not exists(mnist_dir):
                makedirs(mnist_dir)

            # Training data
            plot_mnist_centroids(mnist_train_images, mnist_train_lbls, 'MNIST Training Data Centroids',
                                 mnist_dir + 'train_cent.png')

            # Test data
            plot_mnist_centroids(mnist_test_images, mnist_test_lbls, 'MNIST Test Data Centroids',
                                 mnist_dir + 'test_cent.png')

            # PCA Training data
            plot_2D_data(pca_mnist_train_images, mnist_train_lbls, 'MNIST PCA Training Data',
                         mnist_dir + 'pca_train.png')

            # PCA test data
            plot_2D_data(pca_mnist_test_images, mnist_test_lbls, 'MNIST PCA Test Data',
                         mnist_dir + 'pca_test.png')

            # TSNE Training data
            #plot_2D_data(tsne_mnist_train_images, mnist_train_lbls, 'MNIST TSNE Training Data',
            #             mnist_dir + 'tsne_train.png')

            # TSNE test data
            #plot_2D_data(tsne_mnist_test_images, mnist_test_lbls, 'MNIST TSNE Test Data', mnist_dir + 'tsne_test.png')

            # Classified test data
            if run_nc:
                nc_dir = mnist_dir + 'nc/'
                if not exists(nc_dir):
                    makedirs(nc_dir)

                # PCA data scatterplot
                plot_2D_data(pca_mnist_test_images, pca_nc_mnist_class, 'NC Classified MNIST PCA Test Data',
                             nc_dir + 'pca_nc_class.png')
                # Class mean vectors of classified test data
                plot_mnist_centroids(mnist_test_images, nc_mnist_class, 'NC Classified MNIST Test Data Centroids',
                                     nc_dir + 'nc_class_cent.png')
                plot_mnist_centroids(mnist_test_images, pca_nc_mnist_class,
                                     'NC Classified MNIST PCA Test Data Centroids', nc_dir + 'pca_nc_class_cent.png')


            if run_nsc:
                nsc_dir = mnist_dir + 'nsc/'
                if not exists(nsc_dir):
                    makedirs(nsc_dir)

                # PCA data scatterplots
                subplot_2D_data(pca_mnist_test_images,
                                [pca_nsc_2_mnist_class, pca_nsc_3_mnist_class, pca_nsc_5_mnist_class],
                                'NSC Classified MNIST PCA Test Data', ['2 Subclasses', '3 Subclasses', '5 Subclasses'],
                                nsc_dir + 'pca_nc_class.png')
                # Class mean vectors of classified test data
                plot_mnist_centroids(mnist_test_images, nsc_5_mnist_class, 'NSC Classified MNIST Test Data Centroids',
                                     nsc_dir + 'nc_class_cent.png')
                plot_mnist_centroids(mnist_test_images, pca_nsc_5_mnist_class,
                                     'NSC Classified MNIST PCA Test Data Centroids', nsc_dir + 'pca_nc_class_cent.png')

            if run_nn:
                nn_dir = mnist_dir + 'nn/'
                if not exists(nn_dir):
                    makedirs(nn_dir)

                # PCA data scatterplot
                plot_2D_data(pca_mnist_test_images, pca_nc_mnist_class, 'NN Classified MNIST PCA Test Data',
                             nn_dir + 'pca_nc_class.png')
                # Class mean vectors of classified test data
                plot_mnist_centroids(mnist_test_images, nn_mnist_class, 'NN Classified MNIST Test Data Centroids',
                                     nn_dir + 'nc_class_cent.png')
                plot_mnist_centroids(mnist_test_images, pca_nn_mnist_class,
                                     'NN Classified MNIST PCA Test Data Centroids', nn_dir + 'pca_nc_class_cent.png')

            if run_perc_bp:
                perc_bp_dir = mnist_dir + 'perc-bp/'
                if not exists(perc_bp_dir):
                    makedirs(perc_bp_dir)

                # PCA data scatterplot
                plot_2D_data(pca_mnist_test_images, pca_perc_bp_mnist_class,
                             'Backpropagation Perceptron Classified MNIST PCA Test Data',
                             perc_bp_dir + 'pca_nc_class_cent.png')
                plot_mnist_centroids(mnist_test_images, perc_bp_mnist_class,
                                     'Backpropagation Perceptron Classified MNIST Test Data Centroids',
                                     perc_bp_dir + 'pca_nc_class_cent.png')
                plot_mnist_centroids(mnist_test_images, pca_perc_bp_mnist_class,
                                     'Backpropagation Perceptron Classified MNIST PCA Test Data Centroids',
                                     perc_bp_dir + 'pca_nc_class_cent.png')

            if run_perc_mse:
                perc_mse_dir = mnist_dir + 'perc-mse/'
                if not exists(perc_mse_dir):
                    makedirs(perc_mse_dir)

                # PCA data scatterplot
                plot_2D_data(pca_orl_test_images, pca_perc_mse_mnist_class,
                             'MSE Perceptron Classified MNIST PCA Test Data', perc_mse_dir + 'pca_nc_class_cent.png')
                # Class mean vectors of classified test data
                plot_mnist_centroids(orl_test_images, perc_mse_mnist_class,
                                     'MSE Perceptron Classified MNIST Test Data Centroids',
                                     perc_mse_dir + 'pca_nc_class_cent.png')
                plot_mnist_centroids(orl_test_images, pca_perc_mse_mnist_class,
                                     'MSE Perceptron Classified MNIST PCA Test Data Centroids',
                                     perc_mse_dir + 'pca_nc_class_cent.png')


        """ ********* ORL ********* """
        if run_orl:
            orl_dir = dir + 'orl/'
            if not exists(orl_dir):
                makedirs(orl_dir)

            # Training data
            plot_orl_centroids(orl_train_images, orl_train_lbls, 'ORL Training Data Centroids',
                               orl_dir + 'train_cent.png')

            # Test data
            plot_orl_centroids(orl_test_images, orl_test_lbls, 'ORL Test Data Centroids', orl_dir+'test_cent.png')

            # PCA Training data
            plot_2D_data(pca_orl_train_images, orl_train_lbls, 'ORL PCA Training Data', orl_dir+'pca_train.png')

            # Labeled (actual) PCA test data
            plot_2D_data(pca_orl_test_images, orl_test_lbls, 'ORL PCA Test Data', orl_dir+'pca_test.png')

            # TSNE Training data
            #plot_2D_data(tsne_orl_train_images, orl_train_lbls, 'ORL TSNE Training Data', orl_dir+'tsne_train.png')

            # TSNE test data
            #plot_2D_data(tsne_orl_test_images, orl_test_lbls, 'ORL TSNE Test Data', orl_dir+'tsne_test.png')

            # Classified test data
            if run_nc:
                nc_dir = orl_dir+'nc/'
                if not exists(nc_dir):
                    makedirs(nc_dir)

                # PCA data scatterplot
                plot_2D_data(pca_orl_test_images, pca_nc_orl_class, 'NC Classified ORL PCA Test Data',
                             nc_dir + 'pca_nc_class.png')
                # Class mean vectors of classified test data
                plot_orl_centroids(orl_test_images, nc_orl_class, 'NC Classified ORL Test Data Centroids',
                                   nc_dir + 'nc_class_cent.png')
                plot_orl_centroids(orl_test_images, pca_nc_orl_class, 'NC Classified ORL PCA Test Data Centroids',
                                   nc_dir + 'pca_nc_class_cent.png')

            if run_nsc:
                nsc_dir = orl_dir+'nsc/'
                if not exists(nsc_dir):
                    makedirs(nsc_dir)

                # PCA data scatterplots
                subplot_2D_data(pca_orl_test_images, [pca_nsc_2_orl_class, pca_nsc_3_orl_class, pca_nsc_5_orl_class],
                                'NSC Classified ORL PCA Test Data', ['2 Subclasses', '3 Subclasses', '5 subclasses'],
                                nsc_dir + 'pca_nsc_class.png')
                # Class mean vectors of classified test data
                plot_orl_centroids(orl_test_images, nsc_5_orl_class, 'NSC Classified ORL Test Data Centroids',
                                   nsc_dir + 'nsc_class_cent.png')
                plot_orl_centroids(orl_test_images, pca_nsc_5_orl_class, 'NSC Classified ORL PCA Test Data Centroids',
                                   nsc_dir + 'pca_nsc_class_cent.png')

            if run_nn:
                nn_dir = orl_dir+'nn/'
                if not exists(nn_dir):
                    makedirs(nn_dir)

                # PCA data scatterplot
                plot_2D_data(pca_orl_test_images, pca_nn_orl_class, 'NN Classified ORL PCA Test Data',
                             nn_dir + 'pca_nsc_class.png')
                # Class mean vectors of classified test data
                plot_orl_centroids(orl_test_images, nn_orl_class, 'NN Classified ORL Test Data Centroids',
                                   nn_dir + 'nsc_class_cent.png')
                plot_orl_centroids(orl_test_images, pca_nn_orl_class, 'NN Classified ORL PCA Test Data Centroids',
                                   nn_dir + 'pca_nsc_class_cent.png')

            if run_perc_bp:
                perc_bp_dir = orl_dir+'perc-bp/'
                if not exists(perc_bp_dir):
                    makedirs(perc_bp_dir)

                # PCA data scatterplot
                plot_2D_data(pca_orl_test_images, pca_perc_bp_orl_class,
                             'Backpropagation Perceptron Classified ORL PCA Test Data',
                             perc_bp_dir + 'pca_nsc_class.png')
                plot_orl_centroids(orl_test_images, perc_bp_orl_class,
                                   'Backpropagation Perceptron Classified ORL Test Data Centroids',
                                   perc_bp_dir + 'nsc_class_cent.png')
                plot_orl_centroids(orl_test_images, pca_perc_bp_orl_class,
                                   'Backpropagation Perceptron Classified ORL PCA Test Data Centroids',
                                   perc_bp_dir + 'pca_nsc_class_cent.png')

            if run_perc_mse:
                perc_mse_dir = orl_dir + 'perc-mse/'
                if not exists(perc_mse_dir):
                    makedirs(perc_mse_dir)

                # PCA data scatterplot
                plot_2D_data(pca_orl_test_images, pca_perc_mse_orl_class,
                             'MSE Perceptron Classified ORL PCA Test Data', perc_mse_dir + 'pca_nsc_class.png')
                # Class mean vectors of classified test data
                plot_orl_centroids(orl_test_images, perc_mse_orl_class,
                                   'MSE Perceptron Classified ORL Test Data Centroids',
                                   perc_mse_dir + 'nsc_class_cent.png')
                plot_orl_centroids(orl_test_images, pca_perc_mse_orl_class,
                                   'MSE Perceptron Classified ORL PCA Test Data Centroids',
                                   perc_mse_dir + 'pca_nsc_class_cent.png')

    """ ********* Classification scores ********* """
    print("*** Classification Scores ***\n")
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

        # Backpropagation Perceptron
        if run_perc_bp:
            print("\tBackpropagation Perceptron: " + str(perc_bp_mnist_score))
            print("\tBackpropagation Perceptron w/ PCA: " + str(pca_perc_bp_mnist_score))

        # MSE Perceptron
        if run_perc_mse:
            print("\tMSE Perceptron: " + str(perc_mse_mnist_score))
            print("\tMSE Perceptron w/ PCA: " + str(pca_perc_mse_mnist_score))

    if run_orl:
        print("*** ORL ***")
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
            print("\tNearest Subclass Centroid (5): " + str(nsc_5_orl_score))
            print("\tNearest Subclass Centroid (5) w/ PCA: " + str(pca_nsc_5_orl_score))

        # Nearest Neighbor
        if run_nn:
            print("\tNearest Neighbor: " + str(nn_orl_score))
            print("\tNearest Neighbor w/ PCA: " + str(pca_nn_orl_score))

        # Backpropagation Perceptron
        if run_perc_bp:
            print("\tBackpropagation Perceptron: " + str(perc_bp_orl_score))
            print("\tBackpropagation Perceptron w/ PCA: " + str(pca_perc_bp_orl_score))

        # MSE Perceptron
        if run_perc_mse:
            print("\tMSE Perceptron: " + str(perc_mse_orl_score))
            print("\tMSE Perceptron w/ PCA: " + str(pca_perc_mse_orl_score))

    # Flush results to stdout
    sys.stdout.flush()

    # Block script to keep figures
    plt.show()

if __name__ == "__main__":
    # set which algorithm to apply for this execution
    run_nc = False
    run_nsc = False
    run_nn = False
    show_figs = True
    run_mnist = False
    run_orl = False
    run_perc_bp = False
    run_perc_mse = False
    cpus = 1
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            print("Usage:")
            print("\topti_project.py [<mnist> <orl>] [<nc> <nsc> <nn> <perc-np> <perc-mse>] [no-figs] [cpus=<int>]\n")
            print('[Optional Parameters]:')
            help_text = [['Description', 'Usage', 'Default'],
                         ['-----------','-----------','-----------'],
                         ['Specify Dataset:', 'mnist, orl', 'both'],
                         ['Specify Algorithm:', 'nc, nsc, nn, perc-bp, perc-mse', 'all'],
                         ['Disable figures:', 'no-figs', 'enabled'],
                         ['CPU cores:', 'cpus=[int]', '1\n']]
            col_width = max(len(word) for row in help_text for word in row) + 2  # padding
            for row in help_text:
                print("".join(word.ljust(col_width) for word in row))

            print("Example:")
            print("\topti_project.py mnist nc nn cpus=2")
            exit(0)
        for arg in sys.argv:
            if arg == 'nc' or arg == 'NC':
                run_nc = True
            elif arg == 'nsc' or arg == 'NSC':
                run_nsc = True
            elif arg == 'nn' or arg == 'NN':
                run_nn = True
            elif arg == 'no-figs':
                show_figs = False
            elif arg == 'mnist' or arg == 'MNIST':
                run_mnist = True
            elif arg == 'orl' or arg == 'ORL':
                run_orl = True
            elif arg == 'perc-bp':
                run_perc_bp = True
            elif arg == 'perc-mse':
                run_perc_mse = True
            elif 'cpus=' in arg:
                cpus = int(arg[arg.find('=')+1:])
    else:
        run_nc = True
        run_nsc = True
        run_nn = True
        run_mnist = True
        run_orl = True
        run_perc_bp = True
        run_perc_mse = True

    if (not run_mnist) and (not run_orl):
        run_mnist = True
        run_orl = True

    if (not run_nc) and (not run_nsc) and (not run_nn) and (not run_perc_bp) and (not run_perc_mse):
        run_nc = True
        run_nsc = True
        run_nn = True
        run_perc_bp = True
        run_perc_mse = True

    main(run_mnist, run_orl, run_nc, run_nsc, run_nn, run_perc_bp, run_perc_mse, cpus=cpus)