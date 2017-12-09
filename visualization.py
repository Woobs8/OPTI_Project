import os
import sys
from tools import loadMNIST, loadORL, pca, tsne, plot_mnist_centroids, plot_orl_centroids, plot_2D_data, subplot_2D_data, plot_confusion_matrix, plot_decision_boundary, plot_orl_subclass_centroids, plot_mnist_subclass_centroids, plot_mnist, plot_orl, filter_by
from classify import NC, NSC, NN, BP_Perceptron, MSE_Perceptron
import matplotlib.pyplot as plt
import multiprocessing
from os.path import exists
from os import makedirs
import numpy as np


def main(run_mnist=True, run_orl=True, run_nc=True, run_nsc=True, run_nn=True, run_perc_bp=True, run_perc_mse=True, cpus=1, show_figs=False):
    print("!!!!! Running Classifier Visualization !!!!!", flush=True)

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
            mnist_nc = NC()
            # 784D data
            mnist_nc.fit(mnist_train_images, mnist_train_lbls)
            nc_mnist_class, nc_mnist_score = mnist_nc.predict(mnist_test_images, mnist_test_lbls)
            # PCA data
            mnist_nc.fit(pca_mnist_train_images, mnist_train_lbls)
            pca_nc_mnist_class, pca_nc_mnist_score = mnist_nc.predict(pca_mnist_test_images, mnist_test_lbls)

        # Nearest Subclass Centroid
        if run_nsc:
            # 2 subclasses, 784D data
            mnist_nsc2 = NSC(2)
            mnist_nsc2.fit(mnist_train_images, mnist_train_lbls)
            nsc_2_mnist_class, nsc_2_mnist_score = mnist_nsc2.predict(mnist_test_images, mnist_test_lbls)
            # 2 subclasses, PCA data
            mnist_nsc2.fit(pca_mnist_train_images, mnist_train_lbls)
            pca_nsc_2_mnist_class, pca_nsc_2_mnist_score = mnist_nsc2.predict(pca_mnist_test_images, mnist_test_lbls)

            # 3 subclasses, 784D data
            mnist_nsc3 = NSC(3)
            mnist_nsc3.fit(mnist_train_images, mnist_train_lbls)
            nsc_3_mnist_class, nsc_3_mnist_score = mnist_nsc3.predict(mnist_test_images, mnist_test_lbls)
            # 3 subclasses, PCA data
            mnist_nsc3.fit(pca_mnist_train_images, mnist_train_lbls)
            pca_nsc_3_mnist_class, pca_nsc_3_mnist_score = mnist_nsc3.predict(pca_mnist_test_images, mnist_test_lbls)

            # 5 subclasses, 784D data
            mnist_nsc5 = NSC(5)
            mnist_nsc5.fit(mnist_train_images, mnist_train_lbls)
            nsc_5_mnist_class, nsc_5_mnist_score = mnist_nsc5.predict(mnist_test_images, mnist_test_lbls)
            mnist_nsc5_centroids = mnist_nsc5.subclass_centers  # For visualization purposes
            # 5 subclasses, PCA data
            mnist_nsc5.fit(pca_mnist_train_images, mnist_train_lbls)
            pca_nsc_5_mnist_class, pca_nsc_5_mnist_score = mnist_nsc5.predict(pca_mnist_test_images, mnist_test_lbls)

        # Nearest Neighbor
        if run_nn:
            mnist_nn = NN(1,'uniform',cpus)
            # 784D data
            mnist_nn.fit(mnist_train_images, mnist_train_lbls)
            nn_mnist_class, nn_mnist_score = mnist_nn.predict(mnist_test_images, mnist_test_lbls, 'hard')
            # PCA data
            mnist_nn.fit(pca_mnist_train_images, mnist_train_lbls)
            pca_nn_mnist_class, pca_nn_mnist_score = mnist_nn.predict(pca_mnist_test_images, mnist_test_lbls, 'hard')

        # Backpropagation Perceptron
        if run_perc_bp:
            mnist_perc_bp = BP_Perceptron()
            # 784D data
            mnist_perc_bp.fit(mnist_train_images, mnist_train_lbls, eta=1, eta_decay=0.01, max_iter=100, annealing=True)
            perc_bp_mnist_class, perc_bp_mnist_score = mnist_perc_bp.predict(mnist_test_images, mnist_test_lbls)

            # PCA data
            mnist_perc_bp.fit(pca_mnist_train_images, mnist_train_lbls, eta=1, eta_decay=0.005, max_iter=100, annealing=True)
            pca_perc_bp_mnist_class, pca_perc_bp_mnist_score = mnist_perc_bp.predict(pca_mnist_test_images, mnist_test_lbls)

        # MSE Perceptron
        if run_perc_mse:
            mnist_perc_mse = MSE_Perceptron()
            # 784D data
            mnist_perc_mse.fit(mnist_train_images, mnist_train_lbls, epsilon=10 ** 2)
            perc_mse_mnist_class, perc_mse_mnist_score = mnist_perc_mse.predict(mnist_test_images, mnist_test_lbls)

            # PCA data
            mnist_perc_mse.fit(pca_mnist_train_images, mnist_train_lbls, epsilon=10 ** 2)
            pca_perc_mse_mnist_class, pca_perc_mse_mnist_score = mnist_perc_mse.predict(pca_mnist_test_images, mnist_test_lbls)


    """ ********* Classifying ORL samples ********* """
    if run_orl:
        # Nearest Centroid
        if run_nc:
            orl_nc = NC()
            # 1200D data
            orl_nc.fit(orl_train_images, orl_train_lbls)
            nc_orl_class, nc_orl_score = orl_nc.predict(orl_test_images, orl_test_lbls)
            # PCA data
            orl_nc.fit(pca_orl_train_images, orl_train_lbls)
            pca_nc_orl_class, pca_nc_orl_score = orl_nc.predict(pca_orl_test_images, orl_test_lbls)

        # Nearest Subclass Centroid
        if run_nsc:
            orl_nsc2 = NSC(2)
            # 2 subclasses, 1200D data
            orl_nsc2.fit(orl_train_images, orl_train_lbls)
            nsc_2_orl_class, nsc_2_orl_score = orl_nsc2.predict(orl_test_images, orl_test_lbls)
            # 2 subclasses, PCA data
            orl_nsc2.fit(pca_orl_train_images, orl_train_lbls)
            pca_nsc_2_orl_class, pca_nsc_2_orl_score = orl_nsc2.predict(pca_orl_test_images, orl_test_lbls)

            # 3 subclasses, 1200D data
            orl_nsc3 = NSC(3)
            orl_nsc3.fit(orl_train_images, orl_train_lbls)
            nsc_3_orl_class, nsc_3_orl_score = orl_nsc3.predict(orl_test_images, orl_test_lbls)
            # 3 subclasses, PCA data
            orl_nsc3.fit(pca_orl_train_images, orl_train_lbls)
            pca_nsc_3_orl_class, pca_nsc_3_orl_score = orl_nsc3.predict(pca_orl_test_images, orl_test_lbls)

            # 5 subclasses, 1200D data
            orl_nsc5 = NSC(5)
            orl_nsc5.fit(orl_train_images, orl_train_lbls)
            nsc_5_orl_class, nsc_5_orl_score = orl_nsc5.predict(orl_test_images, orl_test_lbls)
            orl_nsc5_centroids = orl_nsc5.subclass_centers  # For visualization purposes
            # 5 subclasses, PCA data
            orl_nsc5.fit(pca_orl_train_images, orl_train_lbls)
            pca_nsc_5_orl_class, pca_nsc_5_orl_score = orl_nsc5.predict(pca_orl_test_images, orl_test_lbls)

        # Nearest Neighbor
        if run_nn:
            orl_nn = NN(1,'uniform',cpus)
            # 1200D data
            orl_nn.fit(orl_train_images, orl_train_lbls)
            nn_orl_class, nn_orl_score = orl_nn.predict(orl_test_images, orl_test_lbls, 'hard')
            # PCA data
            orl_nn.fit(pca_orl_train_images, orl_train_lbls)
            pca_nn_orl_class, pca_nn_orl_score = orl_nn.predict(pca_orl_test_images, orl_test_lbls, 'hard')

        # Backpropagation Perceptron
        if run_perc_bp:
            orl_perc_bp = BP_Perceptron()
            # 1200D data
            orl_perc_bp.fit(orl_train_images, orl_train_lbls, eta=1, eta_decay=0.006, max_iter=100, annealing=True)
            perc_bp_orl_class, perc_bp_orl_score = orl_perc_bp.predict(orl_test_images, orl_test_lbls)

            # PCA data
            orl_perc_bp.fit(pca_orl_train_images, orl_train_lbls, eta=1, eta_decay=0.07, max_iter=100, annealing=True)
            pca_perc_bp_orl_class, pca_perc_bp_orl_score = orl_perc_bp.predict(pca_orl_test_images, orl_test_lbls)


        # MSE Perceptron
        if run_perc_mse:
            orl_perc_mse = MSE_Perceptron()
            # 1200D data
            orl_perc_mse.fit(orl_train_images, orl_train_lbls, epsilon=10 ** 2)
            perc_mse_orl_class, perc_mse_orl_score = orl_perc_mse.predict(orl_test_images, orl_test_lbls)

            # PCA data
            orl_perc_mse.fit(pca_orl_train_images, orl_train_lbls, epsilon=10 ** 2)
            pca_perc_mse_orl_class, pca_perc_mse_orl_score = orl_perc_mse.predict(pca_orl_test_images, orl_test_lbls)


    """ ********* Data Visualization ********* """
    dir = 'figures/'
    if not exists(dir):
        makedirs(dir)

    """ ********* MNIST ********* """
    if run_mnist:
        mnist_dir = dir + 'mnist/'
        if not exists(mnist_dir):
            makedirs(mnist_dir)

        # Sample data
        plot_mnist(mnist_train_images[:35],'',mnist_dir+'mnist_sample.png', draw=show_figs, columns=7)

        # Sample data for specific class
        class9_samples = filter_by(mnist_train_images, mnist_train_lbls, 9)
        plot_mnist(class9_samples[5:12],fp=mnist_dir+'mnist_9_sample.png', draw=show_figs, columns=7)

        # Training data
        plot_mnist_centroids(mnist_train_images, mnist_train_lbls, 'MNIST Training Data Centroids',
                             mnist_dir + 'train_cent.png', draw=show_figs)

        # Test data
        plot_mnist_centroids(mnist_test_images, mnist_test_lbls, 'MNIST Test Data Centroids',
                             mnist_dir + 'test_cent.png', draw=show_figs)

        # PCA Training data
        plot_2D_data(pca_mnist_train_images, mnist_train_lbls, 'PCA MNIST Training Data',
                     mnist_dir + 'pca_train.png', draw=show_figs)

        # PCA test data
        plot_2D_data(pca_mnist_test_images, mnist_test_lbls, 'PCA MNIST Test Data',
                     mnist_dir + 'pca_test.png', draw=show_figs)

        # TSNE Training data
        #plot_2D_data(tsne_mnist_train_images, mnist_train_lbls, 'TSNE MNIST Training Data',
        #             mnist_dir + 'tsne_train.png')

        # TSNE test data
        #plot_2D_data(tsne_mnist_test_images, mnist_test_lbls, 'TSNE MNIST Test Data', mnist_dir + 'tsne_test.png')

        # Classified test data
        if run_nc:
            nc_dir = mnist_dir + 'nc/'
            if not exists(nc_dir):
                makedirs(nc_dir)

            # PCA data scatterplot
            plot_2D_data(pca_mnist_test_images, pca_nc_mnist_class, 'NC PCA MNIST Test Data',
                         nc_dir + 'pca_nc_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(nc_mnist_class, mnist_test_lbls, True,
                                  'NC MNIST Confusion Matrix', nc_dir + 'nc_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_nc_mnist_class, mnist_test_lbls, True,
                                  'NC PCA MNIST Confusion Matrix',
                                  nc_dir + 'pca_nc_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_mnist_centroids(mnist_test_images, nc_mnist_class, 'NC MNIST Classified Centroids',
                                 nc_dir + 'nc_class_cent.png', draw=show_figs)
            plot_mnist_centroids(mnist_test_images, pca_nc_mnist_class,
                                 'NC PCA MNIST Classified Centroids', nc_dir + 'pca_nc_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(mnist_nc, pca_mnist_test_images, mnist_test_lbls,
                                   'NC PCA MNIST Decision Boundaries', fp=nc_dir + 'pca_nc_dec_bounds.png', draw=show_figs)


        if run_nsc:
            nsc_dir = mnist_dir + 'nsc/'
            if not exists(nsc_dir):
                makedirs(nsc_dir)

            # PCA data scatterplots
            subplot_2D_data(pca_mnist_test_images,
                            [pca_nsc_2_mnist_class, pca_nsc_3_mnist_class, pca_nsc_5_mnist_class],
                            'NSC PCA MNIST Test Data', ['2 Subclasses', '3 Subclasses', '5 Subclasses'],
                            nsc_dir + 'pca_nsc_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(nsc_5_mnist_class, mnist_test_lbls, True,
                                  'NSC MNIST Confusion Matrix', nsc_dir + 'nsc_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_nsc_5_mnist_class, mnist_test_lbls, True,
                                  'NSC PCA MNIST Confusion Matrix',
                                  nsc_dir + 'pca_nsc_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_mnist_centroids(mnist_test_images, nsc_5_mnist_class, 'NSC MNIST Classified Centroids',
                                 nsc_dir + 'nsc_class_cent.png', draw=show_figs)
            plot_mnist_centroids(mnist_test_images, pca_nsc_5_mnist_class,
                                 'NSC PCA MNIST PCA Classified Centroids', nsc_dir + 'pca_nsc_class_cent.png', draw=show_figs)
            # Plot decision boundaries
            plot_decision_boundary(mnist_nsc2, pca_mnist_test_images, mnist_test_lbls,
                                   'NSC(2) PCA MNIST Decision Boundaries',
                                   fp=nsc_dir + 'pca_nsc2_dec_bounds.png', draw=show_figs)
            plot_decision_boundary(mnist_nsc3, pca_mnist_test_images, mnist_test_lbls,
                                   'NSC(3) PCA MNIST Decision Boundaries',
                                   fp=nsc_dir + 'pca_nsc3_dec_bounds.png', draw=show_figs)
            plot_decision_boundary(mnist_nsc5, pca_mnist_test_images, mnist_test_lbls,
                                   'NSC(5) PCA MNIST Decision Boundaries',
                                   fp=nsc_dir + 'pca_nsc5_dec_bounds.png', draw=show_figs)
            # Plot the centroids of the clustered subclasses
            plot_mnist_subclass_centroids(mnist_nsc5_centroids[9],"MNIST(9) Subclasses", nsc_dir + 'nsc_subclasses.png', draw=show_figs)

        if run_nn:
            nn_dir = mnist_dir + 'nn/'
            if not exists(nn_dir):
                makedirs(nn_dir)

            # PCA data scatterplot
            plot_2D_data(pca_mnist_test_images, pca_nn_mnist_class, 'NN PCA MNIST Test Data',
                         nn_dir + 'pca_nn_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(nn_mnist_class, mnist_test_lbls, True,
                                  'NN MNIST Confusion Matrix', nn_dir + 'nn_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_nn_mnist_class, mnist_test_lbls, True,
                                  'NN PCA MNIST Confusion Matrix',
                                  nn_dir + 'pca_nn_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_mnist_centroids(mnist_test_images, nn_mnist_class, 'NN MNIST Classified Centroids',
                                 nn_dir + 'nn_class_cent.png', draw=show_figs)
            plot_mnist_centroids(mnist_test_images, pca_nn_mnist_class,
                                 'NN PCA MNIST Classified Centroids', nn_dir + 'pca_nn_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(mnist_nn, pca_mnist_test_images, mnist_test_lbls,
                                   'NN PCA MNIST Decision Boundaries', fp=nn_dir + 'pca_nn_dec_bounds.png', draw=show_figs)

        if run_perc_bp:
            perc_bp_dir = mnist_dir + 'perc-bp/'
            if not exists(perc_bp_dir):
                makedirs(perc_bp_dir)

            # PCA data scatterplot
            plot_2D_data(pca_mnist_test_images, pca_perc_bp_mnist_class,
                         'BP Perceptron PCA MNIST Test Data',
                         perc_bp_dir + 'pca_perc_bp_class_cent.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(perc_bp_mnist_class, mnist_test_lbls, True,
                                  'BP Perceptron MNIST Confusion Matrix',
                                  perc_bp_dir + 'perc_bp_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_perc_bp_mnist_class, mnist_test_lbls, True,
                                  'BP Perceptron PCA MNIST Confusion Matrix',
                                  perc_bp_dir + 'pca_perc_bp_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_mnist_centroids(mnist_test_images, perc_bp_mnist_class,
                                 'BP Perceptron MNIST Classified Centroids',
                                 perc_bp_dir + 'pca_perc_bp_class_cent.png', draw=show_figs)
            plot_mnist_centroids(mnist_test_images, pca_perc_bp_mnist_class,
                                 'BP Perceptron PCA MNIST Classified Centroids',
                                 perc_bp_dir + 'pca_perc_bp_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(mnist_perc_bp, pca_mnist_test_images, mnist_test_lbls,
                                   'BP Perceptron PCA MNIST Decision Boundaries',
                                   fp=perc_bp_dir + 'pca_perc_bp_dec_bounds.png', draw=show_figs)

        if run_perc_mse:
            perc_mse_dir = mnist_dir + 'perc-mse/'
            if not exists(perc_mse_dir):
                makedirs(perc_mse_dir)

            # PCA data scatterplot
            plot_2D_data(mnist_test_images, pca_perc_mse_mnist_class,
                         'MSE Perceptron PCA MNIST Test Data', perc_mse_dir + 'pca_perc_mse_class_cent.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(perc_mse_mnist_class, mnist_test_lbls, True,
                                  'MSE Perceptron MNIST Confusion Matrix',
                                  perc_mse_dir + 'perc_mse_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_perc_mse_mnist_class, mnist_test_lbls, True,
                                  'MSE Perceptron PCA MNIST Classified Confusion Matrix',
                                  perc_mse_dir + 'pca_perc_mse_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_mnist_centroids(mnist_test_images, perc_mse_mnist_class,
                                 'MSE Perceptron MNIST Classified Centroids',
                                 perc_mse_dir + 'pca_perc_mse_class_cent.png', draw=show_figs)
            plot_mnist_centroids(mnist_test_images, pca_perc_mse_mnist_class,
                                 'MSE Perceptron PCA MNIST Classified Centroids',
                                 perc_mse_dir + 'pca_perc_mse_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(mnist_perc_mse, pca_mnist_test_images, mnist_test_lbls,
                                   'MSE Perceptron PCA MNIST Decision Boundaries',
                                   fp=perc_mse_dir + 'pca_perc_mse_dec_bounds.png', draw=show_figs)


    """ ********* ORL ********* """
    if run_orl:
        orl_dir = dir + 'orl/'
        if not exists(orl_dir):
            makedirs(orl_dir)

        # Sample data
        plot_orl(orl_train_images[:40],'',orl_dir+'orl_sample.png', draw=show_figs)

        # Sample data for specific class
        class23_samples = filter_by(orl_train_images, orl_train_lbls, 23)
        plot_orl(class23_samples[:7], '', orl_dir + 'orl_23_sample.png', columns=7, draw=show_figs)

        # Training data
        plot_orl_centroids(orl_train_images, orl_train_lbls, 'ORL Training Data Centroids',
                           orl_dir + 'train_cent.png', draw=show_figs)

        # Test data
        plot_orl_centroids(orl_test_images, orl_test_lbls, 'ORL Test Data Centroids', orl_dir+'test_cent.png', draw=show_figs)

        # PCA Training data
        plot_2D_data(pca_orl_train_images, orl_train_lbls, 'PCA ORL Training Data', orl_dir+'pca_train.png', draw=show_figs)

        # Labeled (actual, draw=show_figs) PCA test data
        plot_2D_data(pca_orl_test_images, orl_test_lbls, 'PCA ORL Test Data', orl_dir+'pca_test.png', draw=show_figs)

        # TSNE Training data
        #plot_2D_data(tsne_orl_train_images, orl_train_lbls, 'TSNE ORL Training Data', orl_dir+'tsne_train.png', draw=show_figs)

        # TSNE test data
        #plot_2D_data(tsne_orl_test_images, orl_test_lbls, 'TSNE ORL Test Data', orl_dir+'tsne_test.png', draw=show_figs)

        # Classified test data
        if run_nc:
            nc_dir = orl_dir+'nc/'
            if not exists(nc_dir):
                makedirs(nc_dir)

            # PCA data scatterplot
            plot_2D_data(pca_orl_test_images, pca_nc_orl_class, 'NC PCA ORL Test Data',
                         nc_dir + 'pca_nc_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(nc_orl_class, orl_test_lbls, True,
                                  'NC ORL Confusion Matrix', nc_dir + 'nc_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_nc_orl_class, orl_test_lbls, True,
                                  'NC PCA ORL Confusion Matrix',
                                  nc_dir + 'pca_nc_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_orl_centroids(orl_test_images, nc_orl_class, 'NC ORL Classified Centroids',
                               nc_dir + 'nc_class_cent.png', draw=show_figs)
            plot_orl_centroids(orl_test_images, pca_nc_orl_class, 'NC PCA ORL Classified Centroids',
                               nc_dir + 'pca_nc_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(orl_nc, pca_orl_test_images, orl_test_lbls,
                                   'NC PCA ORL Decision Boundaries', fp=nc_dir + 'pca_nc_dec_bounds.png', draw=show_figs)

        if run_nsc:
            nsc_dir = orl_dir+'nsc/'
            if not exists(nsc_dir):
                makedirs(nsc_dir)

            # PCA data scatterplots
            subplot_2D_data(pca_orl_test_images, [pca_nsc_2_orl_class, pca_nsc_3_orl_class, pca_nsc_5_orl_class],
                            'NSC Classified PCA ORL Test Data', ['2 Subclasses', '3 Subclasses', '5 subclasses'],
                            nsc_dir + 'pca_nsc_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(nsc_5_orl_class, orl_test_lbls, True,
                                  'NSC ORL Confusion Matrix', nsc_dir + 'nsc_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_nsc_5_orl_class, orl_test_lbls, True,
                                  'NSC PCA ORL Confusion Matrix',
                                  nsc_dir + 'pca_nsc_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_orl_centroids(orl_test_images, nsc_5_orl_class, 'NSC ORL Classified Centroids',
                               nsc_dir + 'nsc_class_cent.png', draw=show_figs)
            plot_orl_centroids(orl_test_images, pca_nsc_5_orl_class, 'NSC PCA ORL Classified Centroids',
                               nsc_dir + 'pca_nsc_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(orl_nsc2, pca_orl_test_images, orl_test_lbls,
                                   'NSC(2) PCA ORL Decision Boundaries',
                                   fp=nsc_dir + 'pca_nsc2_dec_bounds.png', draw=show_figs)
            plot_decision_boundary(orl_nsc3, pca_orl_test_images, orl_test_lbls,
                                   'NSC(3) PCA ORL Decision Boundaries',
                                   fp=nsc_dir + 'pca_nsc3_dec_bounds.png', draw=show_figs)
            plot_decision_boundary(orl_nsc5, pca_orl_test_images, orl_test_lbls,
                                   'NSC(5) PCA ORL Decision Boundaries',
                                   fp=nsc_dir + 'pca_nsc5_dec_bounds.png', draw=show_figs)
            # Plot the centroids of the clustered subclasses
            plot_orl_subclass_centroids(orl_nsc5_centroids[22],"ORL(23) Subclasses", nsc_dir + 'nsc_subclasses.png', draw=show_figs)

        if run_nn:
            nn_dir = orl_dir+'nn/'
            if not exists(nn_dir):
                makedirs(nn_dir)

            # PCA data scatterplot
            plot_2D_data(pca_orl_test_images, pca_nn_orl_class, 'NN PCA ORL Test Data',
                         nn_dir + 'pca_nn_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(nn_orl_class, orl_test_lbls, True,
                                  'NN ORL Confusion Matrix', nn_dir + 'nn_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_nn_orl_class, orl_test_lbls, True,
                                  'NN PCA ORL Confusion Matrix',
                                  nn_dir + 'pca_nn_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_orl_centroids(orl_test_images, nn_orl_class, 'NN ORL Classified Centroids',
                               nn_dir + 'nn_class_cent.png', draw=show_figs)
            plot_orl_centroids(orl_test_images, pca_nn_orl_class, 'NN PCA ORL Classified Centroids',
                               nn_dir + 'pca_nn_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(orl_nn, pca_orl_test_images, orl_test_lbls,
                                   'NN PCA ORL Decision Boundaries', fp=nn_dir + 'pca_nn_dec_bounds.png', draw=show_figs)

        if run_perc_bp:
            perc_bp_dir = orl_dir+'perc-bp/'
            if not exists(perc_bp_dir):
                makedirs(perc_bp_dir)

            # PCA data scatterplot
            plot_2D_data(pca_orl_test_images, pca_perc_bp_orl_class,
                         'BP Perceptron Classified PCA ORL Test Data',
                         perc_bp_dir + 'pca_perc_bp_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(perc_bp_orl_class, orl_test_lbls, True,
                                  'BP Perceptron ORL Confusion Matrix',
                                  perc_bp_dir + 'perc_bp_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_perc_bp_orl_class, orl_test_lbls, True,
                                  'BP Perceptron PCA ORL Confusion Matrix',
                                  perc_bp_dir + 'pca_perc_bp_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_orl_centroids(orl_test_images, perc_bp_orl_class,
                               'BP Perceptron ORL Classified Centroids',
                               perc_bp_dir + 'perc_bp_class_cent.png', draw=show_figs)
            plot_orl_centroids(orl_test_images, pca_perc_bp_orl_class,
                               'BP Perceptron PCA ORL Classified Centroids',
                               perc_bp_dir + 'pca_perc_bp_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(orl_perc_bp, pca_orl_test_images, orl_test_lbls,
                                   'BP Perceptron PCA ORL Decision Boundaries',
                                   fp=perc_bp_dir + 'pca_perc_bp_dec_bounds.png', draw=show_figs)

        if run_perc_mse:
            perc_mse_dir = orl_dir + 'perc-mse/'
            if not exists(perc_mse_dir):
                makedirs(perc_mse_dir)

            # PCA data scatterplot
            plot_2D_data(pca_orl_test_images, pca_perc_mse_orl_class,
                         'MSE Perceptron PCA ORL Test Data', perc_mse_dir + 'pca_perc_mse_class.png', draw=show_figs)
            # Confusion matrix of classified test data
            plot_confusion_matrix(perc_mse_orl_class, orl_test_lbls, True,
                                  'MSE Perceptron ORL Confusion Matrix',
                                  perc_mse_dir + 'perc_mse_conf_mat.png', draw=show_figs)
            plot_confusion_matrix(pca_perc_mse_orl_class, orl_test_lbls, True,
                                  'MSE Perceptron PCA ORL Confusion Matrix',
                                  perc_mse_dir + 'pca_perc_mse_conf_mat.png', draw=show_figs)
            # Class mean vectors of classified test data
            plot_orl_centroids(orl_test_images, perc_mse_orl_class,
                               'MSE Perceptron ORL Classified Centroids',
                               perc_mse_dir + 'perc_mse_class_cent.png', draw=show_figs)
            plot_orl_centroids(orl_test_images, pca_perc_mse_orl_class,
                               'MSE Perceptron PCA ORL Classified Centroids',
                               perc_mse_dir + 'pca_perc_mse_class_cent.png', draw=show_figs)
            # Plot decision boundary
            plot_decision_boundary(orl_perc_mse, pca_orl_test_images, orl_test_lbls,
                                   'MSE Perceptron PCA ORL Decision Boundaries',
                                   fp=perc_mse_dir + 'pca_perc_mse_dec_bounds.png', draw=show_figs)

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
    if show_figs:
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
        show_figs = False

    main(run_mnist, run_orl, run_nc, run_nsc, run_nn, run_perc_bp, run_perc_mse, cpus=cpus, show_figs=show_figs)