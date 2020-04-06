
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from scipy import sparse
from Gen_RVI_iteration import run_one_random_GRVI
from util import set_seed
from create_mask_network_sparse_equal import create_mask_network_sparse_equal
from sample_initial_parameters import sample_initial_parameters_model



def run_GRVI(args, param_set):

    restart_count = args.restart_count
    k = args.K
    Test_ratio = args.Test_ratio
    network = args.network

    network_file_name = '../Datasets/' + network + '.npz'
    B = sparse.load_npz(network_file_name)

    n = B.shape[1]
    Num_edges = B.count_nonzero()
    print('Number of nodes is {} and number of edges is {}'.format(n, Num_edges))
    Test = Test_ratio * Num_edges
    Train = Test


    for seed in range(restart_count):

        # set seed for both np and tensorflow
        set_seed(seed)

        [test_pairs, train_pairs] = create_mask_network_sparse_equal(B, Test, Train)

        model = sample_initial_parameters_model(k, n)

        run_one_random_GRVI(B, args, param_set, test_pairs, train_pairs, model, Train)

        # curr_model = run_one_random_GRVI(B, args, param_set, test_pairs, train_pairs, model)

        # Save the best model and take its current initial values of tau, lambda_bar, phi_bar

    # Reset the average gradient and run the model till convergence