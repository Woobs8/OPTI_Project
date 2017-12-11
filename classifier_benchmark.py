import os
import sys
from tools import loadMNIST, loadORL, pca, new_train_test_split, plot_classifier_boxplot
from classify import NC, NSC, NN, BP_Perceptron, MSE_Perceptron
import matplotlib.pyplot as plt
import multiprocessing
from os.path import exists
from os import makedirs
import numpy as np


def main(run_mnist=True, run_orl=True, cpus=1, iter=1, file=None):
    print("!!!!! Running Classifier Benchmarking !!!!!", flush=True)
    print("Iterations: {}".format(iter), flush=True)

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
        orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls = loadORL(ORL_PATH)
        # Apply PCA to ORL samples
        pca_orl_train_images, pca_orl_test_images = pca(orl_train_images, orl_test_images)


    """ ********* Performance Parameters ********* """
    # Set parameters for parallel execution
    avail_cpus = multiprocessing.cpu_count()
    print("Utilizing " + str(cpus) + "/" + str(avail_cpus) + " CPU cores for parallel execution.")


    """ ********* Benchmarking classification algorithms for MNIST data ********* """
    classifier_count = 7
    if run_mnist:
        mnist_scores = np.zeros((iter,classifier_count), dtype=float)
        mnist_pca_scores = np.zeros((iter,classifier_count), dtype=float)

        for i in range(iter):
            # Nearest Centroid
            mnist_nc = NC()
            # 784D data
            mnist_nc.fit(mnist_train_images, mnist_train_lbls)
            classification, mnist_scores[i,0] = mnist_nc.predict(mnist_test_images, mnist_test_lbls)
            # PCA data
            mnist_nc.fit(pca_mnist_train_images, mnist_train_lbls)
            classification, mnist_pca_scores[i,0] = mnist_nc.predict(pca_mnist_test_images, mnist_test_lbls)

            # Nearest Subclass Centroid
            subclass_set = [2,3,5]
            for j, subclass_count in enumerate(subclass_set):
                mnist_nsc = NSC(subclass_count)
                mnist_nsc.fit(mnist_train_images, mnist_train_lbls)
                # 784D data
                classification, mnist_scores[i,j+1] = mnist_nsc.predict(mnist_test_images, mnist_test_lbls)
                # PCA data
                mnist_nsc.fit(pca_mnist_train_images, mnist_train_lbls)
                classification, mnist_pca_scores[i,j+1] = mnist_nsc.predict(pca_mnist_test_images, mnist_test_lbls)

            # Nearest Neighbor
            mnist_nn = NN(1, 'distance', cpus)
            # 784D data
            mnist_nn.fit(mnist_train_images, mnist_train_lbls)
            classification, mnist_scores[i,4] = mnist_nn.predict(mnist_test_images, mnist_test_lbls, 'hard')
            # PCA data
            mnist_nn.fit(pca_mnist_train_images, mnist_train_lbls)
            classification, mnist_pca_scores[i,4] = mnist_nn.predict(pca_mnist_test_images, mnist_test_lbls,'hard')

            # Backpropagation Perceptron
            mnist_perc_bp = BP_Perceptron()
            # 784D data
            mnist_perc_bp.fit(mnist_train_images, mnist_train_lbls, eta=1, eta_decay=0.01, max_iter=100, annealing=True)
            classification, mnist_scores[i,5] = mnist_perc_bp.predict(mnist_test_images, mnist_test_lbls)
            # PCA data
            mnist_perc_bp.fit(pca_mnist_train_images, mnist_train_lbls, eta=1, eta_decay=0.005, max_iter=100, annealing=True)
            classification, mnist_pca_scores[i,5] = mnist_perc_bp.predict(pca_mnist_test_images,mnist_test_lbls)

            # MSE Perceptron
            epsilon = 10**2
            mnist_perc_mse = MSE_Perceptron()
            # 784D data
            mnist_perc_mse.fit(mnist_train_images, mnist_train_lbls, epsilon=epsilon)
            classification, mnist_scores[i,6] = mnist_perc_mse.predict(mnist_test_images, mnist_test_lbls)

            # PCA data
            mnist_perc_mse.fit(pca_mnist_train_images, mnist_train_lbls, epsilon=epsilon)
            classification, mnist_pca_scores[i,6] = mnist_perc_mse.predict(pca_mnist_test_images,mnist_test_lbls)

            # Create new training/test split for next iteration
            mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls = new_train_test_split(
                mnist_train_images, mnist_train_lbls, mnist_test_images, mnist_test_lbls, test_size=0.14)
            pca_mnist_train_images, pca_mnist_test_images = pca(mnist_train_images, mnist_test_images)


    """ ********* Benchmarking classification algorithms for ORL data ********* """
    if run_orl:
        orl_scores = np.zeros((iter,classifier_count), dtype=float)
        orl_pca_scores = np.zeros((iter,classifier_count), dtype=float)

        for i in range(iter):
            # Nearest Centroid
            orl_nc = NC()
            # 1200D data
            orl_nc.fit(orl_train_images, orl_train_lbls)
            classification, orl_scores[i,0] = orl_nc.predict(orl_test_images, orl_test_lbls)
            # PCA data
            orl_nc.fit(pca_orl_train_images, orl_train_lbls)
            classification, orl_pca_scores[i,0] = orl_nc.predict(pca_orl_test_images, orl_test_lbls)

            # Nearest Subclass Centroid
            subclass_set = [2,3,5]
            for j, subclass_count in enumerate(subclass_set):
                orl_nsc = NSC(subclass_count)
                # 2 subclasses, 1200D data
                orl_nsc.fit(orl_train_images, orl_train_lbls)
                classification, orl_scores[i,j+1] = orl_nsc.predict(orl_test_images, orl_test_lbls)
                # 2 subclasses, PCA data
                orl_nsc.fit(pca_orl_train_images, orl_train_lbls)
                classification, orl_pca_scores[i,j+1] = orl_nsc.predict(pca_orl_test_images, orl_test_lbls)

            # Nearest Neighbor
            orl_nn = NN(1, 'distance', cpus)
            # 1200D data
            orl_nn.fit(orl_train_images, orl_train_lbls)
            classification, orl_scores[i,4] = orl_nn.predict(orl_test_images, orl_test_lbls, 'hard')
            # PCA data
            orl_nn.fit(pca_orl_train_images, orl_train_lbls)
            classification, orl_pca_scores[i,4] = orl_nn.predict(pca_orl_test_images, orl_test_lbls, 'hard')

            # Backpropagation Perceptron
            orl_perc_bp = BP_Perceptron()
            # 1200D data
            orl_perc_bp.fit(orl_train_images, orl_train_lbls, eta=1, eta_decay=0.006, max_iter=100, annealing=True)
            classification, orl_scores[i,5] = orl_perc_bp.predict(orl_test_images, orl_test_lbls)

            # PCA data
            orl_perc_bp.fit(pca_orl_train_images, orl_train_lbls, eta=1, eta_decay=0.07, max_iter=100, annealing=True)
            classification, orl_pca_scores[i,5] = orl_perc_bp.predict(pca_orl_test_images, orl_test_lbls)

            # MSE Perceptron
            epsilon = 10**2
            orl_perc_mse = MSE_Perceptron()
            # 1200D data
            orl_perc_mse.fit(orl_train_images, orl_train_lbls, epsilon=epsilon)
            classification, orl_scores[i,6] = orl_perc_mse.predict(orl_test_images, orl_test_lbls)

            # PCA data
            orl_perc_mse.fit(pca_orl_train_images, orl_train_lbls, epsilon=epsilon)
            classification, orl_pca_scores[i,6] = orl_perc_mse.predict(pca_orl_test_images,orl_test_lbls)

            # Create new training/test split for next iteration
            orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls = new_train_test_split(
                orl_train_images, orl_train_lbls, orl_test_images, orl_test_lbls, test_size=0.3)
            pca_orl_train_images, pca_orl_test_images = pca(orl_train_images, orl_test_images)

    """ ********* Benchmark scores ********* """
    # Redicrect stdout to file if specified
    if file != None:
        orig_stdout = sys.stdout
        f = open(file, 'w')
        sys.stdout = f

    # Prepare figures folder
    dir = 'figures/'
    if not exists(dir):
        makedirs(dir)

    print("*** Benchmark Scores ***")
    classifiers = ['NC', 'NSC(2)', 'NSC(3)', 'NSC(5)', 'NN', 'BP', 'MSE']

    if run_mnist:
        mnist_dir = dir + 'mnist/'
        if not exists(mnist_dir):
            makedirs(mnist_dir)

        plot_classifier_boxplot(mnist_scores, classifiers, fp=mnist_dir + 'mnist_classifier_comp.png')
        plot_classifier_boxplot(mnist_pca_scores, classifiers, fp=mnist_dir + 'mnist_classifier_pca_comp.png')

        # Calculate means of scores of each iteration
        mean_mnist_scores = mnist_scores.mean(axis=0)
        mean_pca_mnist_score = mnist_pca_scores.mean(axis=0)

        # Calculate variance of scores of each iteration
        var_mnist_scores = mnist_scores.var(axis=0)
        var_pca_mnist_score = mnist_pca_scores.var(axis=0)

        # Calculate standard deviation of scores of each iteration
        std_mnist_scores = mnist_scores.std(axis=0)
        std_pca_mnist_score = mnist_pca_scores.std(axis=0)

        print("*** MNIST ***")
        print("\t*** Original Data ***")
        for (clf, mean_score, var_score, std_score) in zip(classifiers, mean_mnist_scores, var_mnist_scores, std_mnist_scores):
            print("\t{0:}: mean={1:.2f}, var={2:.2f}, std={3:.2f}".format(clf, mean_score, var_score, std_score))

        print("\n\t*** PCA Data ***")
        for (clf, mean_score, var_score, std_score) in zip(classifiers, mean_pca_mnist_score, var_pca_mnist_score, std_pca_mnist_score):
            print("\t{0:}: mean={1:.2f}, var={2:.2f}, std={3:.2f}".format(clf, mean_score, var_score, std_score))

    if run_orl:
        orl_dir = dir + 'orl/'
        if not exists(orl_dir):
            makedirs(orl_dir)

        plot_classifier_boxplot(orl_scores, classifiers, fp=orl_dir + 'orl_classifier_comp.png')
        plot_classifier_boxplot(orl_pca_scores, classifiers, fp=orl_dir + 'orl_classifier_pca_comp.png')

        # Calculate means of scores of each iteration
        mean_orl_scores = orl_scores.mean(axis=0)
        mean_pca_orl_score = orl_pca_scores.mean(axis=0)

        # Calculate variance of scores of each iteration
        var_orl_scores = orl_scores.var(axis=0)
        var_pca_orl_score = orl_pca_scores.var(axis=0)

        # Calculate standard deviation of scores of each iteration
        std_orl_scores = orl_scores.std(axis=0)
        std_pca_orl_score = orl_pca_scores.std(axis=0)

        print("*** ORL ***")
        print("\t*** Original Data ***")
        for (clf, mean_score, var_score, std_score) in zip(classifiers, mean_orl_scores, var_orl_scores, std_orl_scores):
            print("\t{0:}: mean={1:.2f}, var={2:.2f}, std={3:.2f}".format(clf, mean_score, var_score, std_score))

        print("\n\t*** PCA Data ***")
        for (clf, mean_score, var_score, std_score) in zip(classifiers, mean_pca_orl_score, var_pca_orl_score, std_pca_orl_score):
            print("\t{0:}: mean={1:.2f}, var={2:.2f}, std={3:.2f}".format(clf, mean_score, var_score, std_score))

    # Flush results to stdout
    sys.stdout.flush()

    # Restore stdout if changed
    if file != None:
        sys.stdout = orig_stdout
        f.close()

    # Block script to keep figures
    plt.show()

if __name__ == "__main__":
    # set which algorithm to apply for this execution
    run_mnist = False
    run_orl = False
    cpus = 1
    file = None
    iter_count = 1
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            print("Usage:")
            print("\tclassifier_benchmark.py [<mnist> <orl>] [cpus=<int>] [file=<string>]\n")
            print('[Optional Parameters]:')
            help_text = [['Description', 'Usage', 'Default'],
                         ['-----------','-----------','-----------'],
                         ['Specify Dataset:', 'mnist, orl', 'both'],
                         ['CPU cores:', 'cpus=[int]', '1'],
                         ['Store results in file:', 'file=[string]', 'None\n']]
            col_width = max(len(word) for row in help_text for word in row) + 2  # padding
            for row in help_text:
                print("".join(word.ljust(col_width) for word in row))

            print("Example:")
            print("\topti_project.py mnist cpus=2 file=clf_benchmark.txt")
            exit(0)
        for arg in sys.argv:
            if arg == 'mnist' or arg == 'MNIST':
                run_mnist = True
            elif arg == 'orl' or arg == 'ORL':
                run_orl = True
            elif 'iter=' in arg:
                iter_count = int(arg[arg.find('iter=') + 5:])
            elif 'cpus=' in arg:
                cpus = int(arg[arg.find('cpus=')+5:])
            elif 'file=' in arg:
                file = arg[arg.find('file=')+5:]
    else:
        run_mnist = True
        run_orl = True

    if (not run_mnist) and (not run_orl):
        run_mnist = True
        run_orl = True

    main(run_mnist, run_orl, cpus=cpus, iter=iter_count, file=file)