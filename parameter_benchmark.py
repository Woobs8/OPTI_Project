import os
import sys
from tools import loadMNIST, loadORL, pca
from classify import NSC, NN, BP_Perceptron, MSE_Perceptron
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np


def main(run_mnist=True, run_orl=True, run_nsc=True, run_nn=True, run_perc_bp=True, run_perc_mse=True, cpus=1, file=None):
    print("!!!!! Running Parameter Benchmarking !!!!!", flush=True)

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


    """ ********* Benchmarking algorithm hyperparameters for MNIST data ********* """
    if run_mnist:
        # Nearest Subclass Centroid
        if run_nsc:
            subclass_set = [2,3,5]
            nsc_mnist_scores = np.zeros((len(subclass_set),3))
            for i, subclass_count in enumerate(subclass_set):
                mnist_nsc = NSC(subclass_count)
                mnist_nsc.fit(mnist_train_images, mnist_train_lbls)
                # 784D data
                classification, nsc_mnist_scores[i,0] = mnist_nsc.predict(mnist_test_images, mnist_test_lbls)
                # PCA data
                mnist_nsc.fit(pca_mnist_train_images, mnist_train_lbls)
                classification, nsc_mnist_scores[i,1] = mnist_nsc.predict(pca_mnist_test_images, mnist_test_lbls)
                nsc_mnist_scores[i,2] = subclass_count

        # Nearest Neighbor
        if run_nn:
            nn_set = range(1,10)
            nn_mnist_scores = np.zeros((len(nn_set),3))
            for i, neighbor_count in enumerate(nn_set):
                mnist_nn = NN(neighbor_count, 'uniform', cpus)
                # 784D data
                mnist_nn.fit(mnist_train_images, mnist_train_lbls)
                classification, nn_mnist_scores[i,0] = mnist_nn.predict(mnist_test_images, mnist_test_lbls, 'hard')
                # PCA data
                mnist_nn.fit(pca_mnist_train_images, mnist_train_lbls)
                classification, nn_mnist_scores[i,1] = mnist_nn.predict(pca_mnist_test_images, mnist_test_lbls,'hard')
                nn_mnist_scores[i,2] = neighbor_count

        # Backpropagation Perceptron
        if run_perc_bp:
            anneal_decay = np.concatenate((np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.06, 0.01)))
            eta_range = np.concatenate((np.arange(0.1, 1.1, 0.1), np.arange(10, 50, 10)))
            perc_bp_mnist_scores = np.zeros((len(eta_range),3))
            perc_bp_anneal_mnist_scores = np.zeros((len(anneal_decay),3))
            mnist_perc_bp = BP_Perceptron()

            for i, (eta, decay) in enumerate(zip(eta_range, anneal_decay)):
                # Without annealing
                # 784D data
                mnist_perc_bp.fit(mnist_train_images, mnist_train_lbls, eta=eta, max_iter=100, annealing=False)
                classification, perc_bp_mnist_scores[i,0] = mnist_perc_bp.predict(mnist_test_images, mnist_test_lbls)
                # PCA data
                mnist_perc_bp.fit(pca_mnist_train_images, mnist_train_lbls, eta=eta, max_iter=100, annealing=False)
                classification, perc_bp_mnist_scores[i,1] = mnist_perc_bp.predict(pca_mnist_test_images,mnist_test_lbls)
                perc_bp_mnist_scores[i,2] = eta

                # With annealing
                # 784D data
                mnist_perc_bp.fit(mnist_train_images, mnist_train_lbls, eta=1, eta_decay=decay, max_iter=100, annealing=True)
                classification, perc_bp_anneal_mnist_scores[i,0] = mnist_perc_bp.predict(mnist_test_images, mnist_test_lbls)
                # PCA data
                mnist_perc_bp.fit(pca_mnist_train_images, mnist_train_lbls, eta=1, eta_decay=decay, max_iter=100, annealing=True)
                classification, perc_bp_anneal_mnist_scores[i,1] = mnist_perc_bp.predict(pca_mnist_test_images,mnist_test_lbls)
                perc_bp_anneal_mnist_scores[i,2] = decay


        # MSE Perceptron
        if run_perc_mse:
            epsilon_exponent_range = range(-6,6,1)
            perc_mse_mnist_scores = np.zeros((len(epsilon_exponent_range),3))
            mnist_perc_mse = MSE_Perceptron()
            for i, epsilon_exp in enumerate(epsilon_exponent_range):
                epsilon = 10**epsilon_exp
                # 784D data
                mnist_perc_mse.fit(mnist_train_images, mnist_train_lbls, epsilon=epsilon)
                classification, perc_mse_mnist_scores[i,0] = mnist_perc_mse.predict(mnist_test_images, mnist_test_lbls)

                # PCA data
                mnist_perc_mse.fit(pca_mnist_train_images, mnist_train_lbls, epsilon=epsilon)
                classification, perc_mse_mnist_scores[i,1] = mnist_perc_mse.predict(pca_mnist_test_images,mnist_test_lbls)
                perc_mse_mnist_scores[i,2] = epsilon_exp



    """ ********* Benchmarking algorithms hyperparameters for ORL data ********* """
    if run_orl:
        # Nearest Subclass Centroid
        if run_nsc:
            subclass_set = [2,3,5]
            nsc_orl_scores = np.zeros((len(subclass_set),3))
            for i, subclass_count in enumerate(subclass_set):
                orl_nsc = NSC(subclass_count)
                # 2 subclasses, 1200D data
                orl_nsc.fit(orl_train_images, orl_train_lbls)
                classification, nsc_orl_scores[i,0] = orl_nsc.predict(orl_test_images, orl_test_lbls)
                # 2 subclasses, PCA data
                orl_nsc.fit(pca_orl_train_images, orl_train_lbls)
                classification, nsc_orl_scores[i,1] = orl_nsc.predict(pca_orl_test_images, orl_test_lbls)
                nsc_orl_scores[i,2] = subclass_count

        # Nearest Neighbor
        if run_nn:
            nn_set = range(1,11)
            nn_orl_scores = np.zeros((len(nn_set),3))
            for i, neighbor_count in enumerate(nn_set):
                orl_nn = NN(neighbor_count, 'uniform', cpus)
                # 1200D data
                orl_nn.fit(orl_train_images, orl_train_lbls)
                classification, nn_orl_scores[i,0] = orl_nn.predict(orl_test_images, orl_test_lbls, 'hard')
                # PCA data
                orl_nn.fit(pca_orl_train_images, orl_train_lbls)
                classification, nn_orl_scores[i,1] = orl_nn.predict(pca_orl_test_images, orl_test_lbls, 'hard')
                nn_orl_scores[i,2] = neighbor_count

        # Backpropagation Perceptron
        if run_perc_bp:
            anneal_decay = np.concatenate((np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.06, 0.01)))
            eta_range = np.concatenate((np.arange(0.1, 1.1, 0.1), np.arange(10, 50, 10)))
            perc_bp_orl_scores = np.zeros((len(eta_range),3))
            perc_bp_anneal_orl_scores = np.zeros((len(anneal_decay),3))
            orl_perc_bp = BP_Perceptron()
            for i, (eta, decay) in enumerate(zip(eta_range, anneal_decay)):
                # Without annealing
                # 1200D data
                orl_perc_bp.fit(orl_train_images, orl_train_lbls, eta=eta, max_iter=100, annealing=False)
                classification, perc_bp_orl_scores[i,0] = orl_perc_bp.predict(orl_test_images, orl_test_lbls)

                # PCA data
                orl_perc_bp.fit(pca_orl_train_images, orl_train_lbls, eta=eta, max_iter=100, annealing=False)
                classification, perc_bp_orl_scores[i,1] = orl_perc_bp.predict(pca_orl_test_images, orl_test_lbls)
                perc_bp_orl_scores[i,2] = eta

                # With annealing
                # 1200D data
                orl_perc_bp.fit(orl_train_images, orl_train_lbls, eta=1, eta_decay=decay, max_iter=100, annealing=True)
                classification, perc_bp_anneal_orl_scores[i,0] = orl_perc_bp.predict(orl_test_images, orl_test_lbls)

                # PCA data
                orl_perc_bp.fit(pca_orl_train_images, orl_train_lbls, eta=1, eta_decay=decay, max_iter=100, annealing=True)
                classification, perc_bp_anneal_orl_scores[i,1] = orl_perc_bp.predict(pca_orl_test_images, orl_test_lbls)
                perc_bp_anneal_orl_scores[i,2] = decay

        # MSE Perceptron
        if run_perc_mse:
            epsilon_exponent_range = range(-6,6,1)
            perc_mse_orl_scores = np.zeros((len(epsilon_exponent_range),3))
            orl_perc_mse = MSE_Perceptron()
            for i, epsilon_exp in enumerate(epsilon_exponent_range):
                epsilon = 10**epsilon_exp
                # 1200D data
                orl_perc_mse.fit(orl_train_images, orl_train_lbls, epsilon=epsilon)
                classification, perc_mse_orl_scores[i,0] = orl_perc_mse.predict(orl_test_images, orl_test_lbls)

                # PCA data
                orl_perc_mse.fit(pca_orl_train_images, orl_train_lbls, epsilon=epsilon)
                classification, perc_mse_orl_scores[i,1] = orl_perc_mse.predict(pca_orl_test_images,orl_test_lbls)
                perc_mse_orl_scores[i,2] = epsilon_exp

    """ ********* Benchmark scores ********* """
    # Redicrect stdout to file if specified
    if file != None:
        orig_stdout = sys.stdout
        f = open(file, 'w')
        sys.stdout = f

    print("*** Benchmark Scores ***")
    if run_mnist:
        print("*** MNIST ***")
        # Nearest Subclass Centroid
        if run_nsc:
            print("\n\t*** Nearest Subclass Centroid ***")
            print("\t*** Original Data ***")
            for score in nsc_mnist_scores:
                print("\tNearest Subclass Centroid ({}): {}".format(str(score[0]), score[0]))

            print("\n\t*** PCA Data ***")
            for score in nsc_mnist_scores:
                print("\tNearest Subclass Centroid ({}) w/ PCA: {}".format(int(score[2]), score[1]))
            print("\t-----------------------------------------------------\n")

        # Nearest Neighbor
        if run_nn:
            print("\n\t*** Nearest Neighbor ***")
            print("\t*** Original Data ***")
            for score in nn_mnist_scores:
                print("\tNearest Neighbor ({}): {}".format(int(score[2]),score[0]))

            print("\n\t*** PCA Data ***")
            for score in nn_mnist_scores:
                print("\tNearest Neighbor ({}) w/ PCA: {}".format(int(score[2]),score[1]))
            print("\t-----------------------------------------------------\n")

        # Backpropagation Perceptron
        if run_perc_bp:
            print("\n\t*** Backpropagation Perceptron ***")
            print("\t*** Original Data ***")
            for score in perc_bp_mnist_scores:
                print("\tBackpropagation Perceptron (eta={0:.3f}): {1:}".format(score[2], score[0]))
            print("\n\t*** Original Data w/ Annealing ***")
            for score in perc_bp_anneal_mnist_scores:
                print("\tBackpropagation Perceptron (annealing) (eta=1, k={0:.3f}): {1:}".format(score[2],score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_bp_mnist_scores:
                print("\tBackpropagation Perceptron w/ PCA (eta={0:.3f}): {1:}".format(score[2], score[1]))
            print("\n\t*** PCA Data w/ Annealing ***")
            for score in perc_bp_anneal_mnist_scores:
                print("\tBackpropagation Perceptron (annealing) w/ PCA (eta=1, k={0:.3f}): {1:}".format(score[2],score[1]))
            print("\t-----------------------------------------------------\n")

        # MSE Perceptron
        if run_perc_mse:
            print("\n\t*** MSE Perceptron ***")
            print("\t*** Original Data ***")
            for score in perc_mse_mnist_scores:
                print("\tMSE Perceptron (epsilon=10**{}): {}".format(int(score[2]),score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_mse_mnist_scores:
                print("\tMSE Perceptron (epsilon=10**{}) w/ PCA: {}".format(int(score[2]),score[1]))
            print("\t-----------------------------------------------------\n")

    if run_orl:
        print("*** ORL ***")
        # Nearest Subclass Centroid
        if run_nsc:
            print("\n\t*** Nearest Subclass Centroid ***")
            print("\t*** Original Data ***")
            for score in nsc_orl_scores:
                print("\tNearest Subclass Centroid ({}): {}".format(int(score[2]),score[0]))

            print("\n\t*** PCA Data ***")
            for score in nsc_orl_scores:
                print("\tNearest Subclass Centroid ({}) w/ PCA: {}".format(int(score[2]),score[1]))
            print("\t-----------------------------------------------------\n")

        # Nearest Neighbor
        if run_nn:
            print("\n\t*** Nearest Neighbor ***")
            print("\t*** Original Data ***")
            for score in nn_orl_scores:
                print("\tNearest Neighbor ({}): {}".format(int(score[2]),str(score[0])))

            print("\n\t*** PCA Data ***")
            for score in nn_orl_scores:
                print("\tNearest Neighbor ({}) w/ PCA: {}".format(int(score[2]),str(score[1])))
            print("\t-----------------------------------------------------\n")

        # Backpropagation Perceptron
        if run_perc_bp:
            print("\n\t*** Backpropagation Perceptron ***")
            print("\t*** Original Data ***")
            for score in perc_bp_orl_scores:
                print("\tBackpropagation Perceptron (eta={0:.3f}): {1:}".format(score[2], score[0]))

            print("\n\t*** Original Data w/ Annealing ***")
            for score in perc_bp_anneal_orl_scores:
                print("\tBackpropagation Perceptron (annealing) (eta=1, k={0:.3f}): {1:}".format(score[2], score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_bp_orl_scores:
                print("\tBackpropagation Perceptron w/ PCA (eta={0:.3f}): {1:}".format(score[2], score[1]))

            print("\n\t*** PCA Data w/ Annealing ***")
            for score in perc_bp_anneal_orl_scores:
                print("\tBackpropagation Perceptron (annealing) w/ PCA (eta=1, k={0:.3f}): {1:}".format(score[2], score[1]))
            print("-----------------------------------------------\n")

        # MSE Perceptron
        if run_perc_mse:
            print("\n\t*** MSE Perceptron ***")
            print("\t*** Original Data ***")
            for score in perc_mse_orl_scores:
                print("\tMSE Perceptron (epsilon=10**{}): {}".format(int(score[2]),score[0]))

            print("\n\t*** PCA Data ***")
            for score in perc_mse_orl_scores:
                print("\tMSE Perceptron (epsilon=10**{}) w/ PCA: {}".format(int(score[2]),score[1]))
            print("\t-----------------------------------------------------\n")

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
    run_nsc = False
    run_nn = False
    run_mnist = False
    run_orl = False
    run_perc_bp = False
    run_perc_mse = False
    cpus = 1
    file = None
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            print("Usage:")
            print("\topti_project.py [<mnist> <orl>] [<nsc> <nn> <perc-np> <perc-mse>] [cpus=<int>] [file=<string>]\n")
            print('[Optional Parameters]:')
            help_text = [['Description', 'Usage', 'Default'],
                         ['-----------','-----------','-----------'],
                         ['Specify Dataset:', 'mnist, orl', 'both'],
                         ['Specify Algorithm:', 'nsc, nn, perc-bp, perc-mse', 'all'],
                         ['CPU cores:', 'cpus=[int]', '1'],
                         ['Store results in file:', 'file=[string]', 'None\n']]
            col_width = max(len(word) for row in help_text for word in row) + 2  # padding
            for row in help_text:
                print("".join(word.ljust(col_width) for word in row))

            print("Example:")
            print("\topti_project.py mnist nc nn cpus=2 file=param_benchmark.txt")
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
                cpus = int(arg[arg.find('cpus=')+5:])
            elif 'file=' in arg:
                file = arg[arg.find('file=')+5:]
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

    main(run_mnist, run_orl, run_nsc, run_nn, run_perc_bp, run_perc_mse, cpus=cpus, file=file)