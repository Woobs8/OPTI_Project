import os
import sys
from tools import loadMNIST, loadORL, pca, plot_mnist_centroids, plot_orl_centroids, plot_2D_data, subplot_2D_data
from classify import nsc, nn, perceptron_bp, perceptron_classify, perceptron_mse
import matplotlib.pyplot as plt
import multiprocessing
from os.path import exists
from os import makedirs
import numpy as np


def main(run_mnist=True, run_orl=True, run_nsc=True, run_nn=True, run_perc_bp=True, run_perc_mse=True, cpus=1):

    """ ********* Loading MNIST samples ********* """
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
        orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls = loadORL(ORL_PATH, seed=42)
        # Apply PCA to ORL samples
        pca_orl_train_images, pca_orl_test_images = pca(orl_train_images, orl_test_images)

    """ ********* Performance Parameters ********* """
    # Set parameters for parallel execution
    avail_cpus = multiprocessing.cpu_count()
    print("Utilizing " + str(cpus) + "/" + str(avail_cpus) + " CPU cores for parallel execution.")


    """ ********* Benchmarking classification algorithms for MNIST data ********* """
    if run_mnist:
        # Nearest Subclass Centroid
        if run_nsc:
            subclass_set = [2,3,5]
            nsc_mnist_scores = np.zeros((len(subclass_set),3))
            for i, subclass_count in enumerate(subclass_set):
                # 784D data
                classification, nsc_mnist_scores[i,0] = nsc(mnist_train_images, mnist_train_lbls, mnist_test_images,
                                                       mnist_test_lbls, subclass_count)
                # PCA data
                classification, nsc_mnist_scores[i,1] = nsc(pca_mnist_train_images, mnist_train_lbls, pca_mnist_test_images,
                                                       mnist_test_lbls, subclass_count)
                nsc_mnist_scores[i,2] = subclass_count

        # Nearest Neighbor
        if run_nn:
            nn_set = range(1,11)
            nn_mnist_scores = np.zeros((len(nn_set),3))
            for i, neighbor_count in enumerate(nn_set):
                # 784D data
                classification, nn_mnist_scores[i,0] = nn(mnist_train_images, mnist_train_lbls, mnist_test_images,
                                                    mnist_test_lbls, neighbor_count, 'uniform', cpus, 'hard')
                # PCA data
                classification, nn_mnist_scores[i,1] = nn(pca_mnist_train_images, mnist_train_lbls,
                                                            pca_mnist_test_images, mnist_test_lbls, neighbor_count,
                                                            'uniform', cpus, 'hard')
                nn_mnist_scores[i,2] = neighbor_count

        # Backpropagation Perceptron
        if run_perc_bp:
            eta_exponent_range = range(-6,1,1)
            perc_bp_mnist_scores = np.zeros((len(eta_exponent_range),3))
            for i, eta_exp in enumerate(eta_exponent_range):
                eta = 10**eta_exp
                # 784D data
                W = perceptron_bp(mnist_train_images, mnist_train_lbls, eta=eta, max_iter=80)
                classification, perc_bp_mnist_scores[i,0] = perceptron_classify(W, mnist_test_images,mnist_test_lbls)

                # PCA data
                W = perceptron_bp(pca_mnist_train_images, mnist_train_lbls, eta=eta, max_iter=80)
                classification, perc_bp_mnist_scores[i,1] = perceptron_classify(W, pca_mnist_test_images,mnist_test_lbls)

                perc_bp_mnist_scores[i,2] = eta_exp

        # MSE Perceptron
        if run_perc_mse:
            epsilon_exponent_range = range(-6,6,1)
            perc_mse_mnist_scores = np.zeros((len(epsilon_exponent_range),3))
            for i, epsilon_exp in enumerate(epsilon_exponent_range):
                epsilon = 10**epsilon_exp
                # 1200D data
                W = perceptron_mse(mnist_train_images, mnist_train_lbls, epsilon=epsilon)
                classification, perc_mse_mnist_scores[i,0] = perceptron_classify(W, mnist_test_images, mnist_test_lbls)

                # PCA data
                W = perceptron_mse(pca_mnist_train_images, mnist_train_lbls, epsilon=epsilon)
                classification, perc_mse_mnist_scores[i,1] = perceptron_classify(W, pca_mnist_test_images,
                                                                                         mnist_test_lbls)
                perc_mse_mnist_scores[i,2] = epsilon_exp



    """ ********* Benchmarking classification algorithms for ORL data ********* """
    if run_orl:
        # Nearest Subclass Centroid
        if run_nsc:
            subclass_set = [2,3,5]
            nsc_orl_scores = np.zeros((len(subclass_set),3))
            for i, subclass_count in enumerate(subclass_set):
                # 784D data
                classification, nsc_orl_scores[i,0] = nsc(orl_train_images, orl_train_lbls, orl_test_images,
                                                            orl_test_lbls, subclass_count)
                # PCA data
                classification, nsc_orl_scores[i,1] = nsc(pca_orl_train_images, orl_train_lbls, pca_orl_test_images,
                                                          orl_test_lbls, subclass_count)
                nsc_orl_scores[i,2] = subclass_count

        # Nearest Neighbor
        if run_nn:
            nn_set = range(1,11)
            nn_orl_scores = np.zeros((len(nn_set),3))
            for i, neighbor_count in enumerate(nn_set):
                # 784D data
                classification, nn_orl_scores[i,0] = nn(orl_train_images, orl_train_lbls, orl_test_images,
                                                          orl_test_lbls, neighbor_count, 'uniform', cpus, 'hard')
                # PCA data
                classification, nn_orl_scores[i,1] = nn(pca_orl_train_images, orl_train_lbls,
                                                          pca_orl_test_images, orl_test_lbls, neighbor_count,
                                                            'uniform', cpus, 'hard')
                nn_orl_scores[i,2] = neighbor_count

        # Backpropagation Perceptron
        if run_perc_bp:
            eta_exponent_range = range(-6,1,1)
            perc_bp_orl_scores = np.zeros((len(eta_exponent_range),3))
            for i, eta_exp in enumerate(eta_exponent_range):
                eta = 10**eta_exp
                # 784D data
                W = perceptron_bp(orl_train_images, orl_train_lbls, eta=eta, max_iter=80)
                classification, perc_bp_orl_scores[i,0] = perceptron_classify(W, orl_test_images,orl_test_lbls)

                # PCA data
                W = perceptron_bp(pca_orl_train_images, orl_train_lbls, eta=eta, max_iter=80)
                classification, perc_bp_orl_scores[i,1] = perceptron_classify(W, pca_orl_test_images,orl_test_lbls)

                perc_bp_orl_scores[i,2] = eta_exp

        # MSE Perceptron
        if run_perc_mse:
            epsilon_exponent_range = range(-6,6,1)
            perc_mse_orl_scores = np.zeros((len(epsilon_exponent_range),3))
            for i, epsilon_exp in enumerate(epsilon_exponent_range):
                epsilon = 10**epsilon_exp
                # 1200D data
                W = perceptron_mse(orl_train_images, orl_train_lbls, epsilon=epsilon)
                classification, perc_mse_orl_scores[i,0] = perceptron_classify(W, orl_test_images, orl_test_lbls)

                # PCA data
                W = perceptron_mse(pca_orl_train_images, orl_train_lbls, epsilon=epsilon)
                classification, perc_mse_orl_scores[i,1] = perceptron_classify(W, pca_orl_test_images,
                                                                               orl_test_lbls)
                perc_mse_orl_scores[i,2] = epsilon_exp

    """ ********* Benchmark scores ********* """
    print("*** Benchmark Scores ***")
    if run_mnist:
        print("*** MNIST ***")
        # Nearest Subclass Centroid
        if run_nsc:
            print("\t*** Original Data ***")
            for score in nsc_mnist_scores:
                print("\tNearest Subclass Centroid ({}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in nsc_mnist_scores:
                print("\tNearest Subclass Centroid ({}) w/ PCA: ".format(int(score[2])) + str(score[1]))

        # Nearest Neighbor
        if run_nn:
            print("\t*** Original Data ***")
            for score in nn_mnist_scores:
                print("\tNearest Neighbor ({}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in nn_mnist_scores:
                print("\tNearest Neighbor ({}) w/ PCA: ".format(int(score[2])) + str(score[1]))

        # Backpropagation Perceptron
        if run_perc_bp:
            print("\t*** Original Data ***")
            for score in perc_bp_mnist_scores:
                print("\tBackpropagation Perceptron (eta=10**{}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_bp_mnist_scores:
                print("\tBackpropagation Perceptron (eta=10**{}) w/ PCA: ".format(int(score[2])) + str(score[1]))

        # MSE Perceptron
        if run_perc_mse:
            print("\t*** Original Data ***")
            for score in perc_mse_mnist_scores:
                print("\tMSE Perceptron (epsilon=10**{}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_mse_mnist_scores:
                print("\tMSE Perceptron (epsilon=10**{}) w/ PCA: ".format(int(score[2])) + str(score[1]))

    if run_orl:
        print("*** ORL ***")
        # Nearest Subclass Centroid
        if run_nsc:
            print("\t*** Original Data ***")
            for score in nsc_orl_scores:
                print("\tNearest Subclass Centroid ({}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in nsc_orl_scores:
                print("\tNearest Subclass Centroid ({}) w/ PCA: ".format(int(score[2])) + str(score[1]))

        # Nearest Neighbor
        if run_nn:
            print("\t*** Original Data ***")
            for score in nn_orl_scores:
                print("\tNearest Neighbor ({}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in nn_orl_scores:
                print("\tNearest Neighbor ({}) w/ PCA: ".format(int(score[2])) + str(score[1]))

        # Backpropagation Perceptron
        if run_perc_bp:
            print("\t*** Original Data ***")
            for score in perc_bp_orl_scores:
                print("\tBackpropagation Perceptron (eta=10**{}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_bp_orl_scores:
                print("\tBackpropagation Perceptron (eta=10**{}) w/ PCA: ".format(int(score[2])) + str(score[1]))

        # MSE Perceptron
        if run_perc_mse:
            print("\t*** Original Data ***")
            for score in perc_mse_orl_scores:
                print("\tMSE Perceptron (epsilon=10**{}): ".format(int(score[2])) + str(score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_mse_orl_scores:
                print("\tMSE Perceptron (epsilon=10**{}) w/ PCA: ".format(int(score[2])) + str(score[1]))

    # Flush results to stdout
    sys.stdout.flush()

    # Block script to keep figures
    plt.show()

if __name__ == "__main__":
    # set which algorithm to apply for this execution
    run_nsc = False
    run_nn = False
    run_mnist = False
    run_orl = False
    run_perc_bp = False
    run_perc_mse = False
    cpus = 1
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            print("Usage:")
            print("\topti_project.py [<mnist> <orl>] [<nsc> <nn> <perc-np> <perc-mse>] [cpus=<int>]\n")
            print('[Optional Parameters]:')
            help_text = [['Description', 'Usage', 'Default'],
                         ['-----------','-----------','-----------'],
                         ['Specify Dataset:', 'mnist, orl', 'both'],
                         ['Specify Algorithm:', 'nsc, nn, perc-bp, perc-mse', 'all'],
                         ['CPU cores:', 'cpus=[int]', '1\n']]
            col_width = max(len(word) for row in help_text for word in row) + 2  # padding
            for row in help_text:
                print("".join(word.ljust(col_width) for word in row))

            print("Example:")
            print("\topti_project.py mnist nc nn cpus=2")
            exit(0)
        for arg in sys.argv:
            if arg == 'nsc' or arg == 'NSC':
                run_nsc = True
            elif arg == 'nn' or arg == 'NN':
                run_nn = True
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
        run_nsc = True
        run_nn = True
        run_mnist = True
        run_orl = True
        run_perc_bp = True
        run_perc_mse = True

    if (not run_mnist) and (not run_orl):
        run_mnist = True
        run_orl = True

    if (not run_nsc) and (not run_nn) and (not run_perc_bp) and (not run_perc_mse):
        run_nsc = True
        run_nn = True
        run_perc_bp = True
        run_perc_mse = True

    main(run_mnist, run_orl, run_nsc, run_nn, run_perc_bp, run_perc_mse, cpus=cpus)