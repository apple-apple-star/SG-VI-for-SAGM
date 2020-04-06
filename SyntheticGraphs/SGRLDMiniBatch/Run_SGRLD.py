from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from scipy import sparse
from SGRLD_iteration import run_random_SGRLD
from util import set_seed
from create_mask_network_sparse_equal import create_mask_network_sparse_equal
from sample_initial_parameters import sample_initial_parameters_model

def run_SGRLD(args, param_set):
    k = args.K
    seed = args.seed
    Test_ratio = args.Test_ratio
    network = args.network

    network_file_name = '../Datasets/' + network + '.npz'
    B = sparse.load_npz(network_file_name)

    n = B.shape[1]
    Num_edges = B.count_nonzero()
    print('Number of nodes is {} and number of edges is {}'.format(n, Num_edges))
    Test = Test_ratio * Num_edges
    Train = Test
    Valid = Test
    initial_step_size_range = [1.0, 0.1, 0.01, 0.001]

    ##
    set_seed(seed)
    [test_pairs, train_pairs, valid_pairs] = create_mask_network_sparse_equal(B, Test, Train, Valid)

    model = sample_initial_parameters_model(k, n, param_set.eta_0, param_set.eta_1)

    outfile_name = args.outfile + network + 'Seed' + str(seed) + 'L' + str(args.L) + 'M' + str(args.m) + 'K' + str(k) \
                  + 'burn_in' + str(args.num_burn_in) + 'samples' + str(args.num_samples) + 'test_ratio' \
                  + str(args.Test_ratio) + 'lrw' + str(param_set.step_size_a_w) \
                  + 'lrpi' + str(param_set.step_size_a_pi)

    if param_set.step_size_a_w == 0 or param_set.step_size_a_pi == 0:
        outfile_name = args.outfile + 'LineSearch_' + network + 'Seed' + str(seed) + 'L' + str(args.L) + 'M' + \
                       str(args.m) + 'K' + str(k) + 'burn_in' + str(args.num_burn_in) + 'samples' + \
                       str(args.num_samples) + 'test_ratio' + str(args.Test_ratio) + 'lrw' + str(param_set.step_size_a_w) \
                       + 'lrpi' + str(param_set.step_size_a_pi)
        all_auc = []
        num_burn_in = args.num_burn_in
        args.num_burn_in = 0
        num_samples = args.num_samples
        args.num_samples = 2000

        for i in initial_step_size_range:  # w
            for j in initial_step_size_range:  # pi
                    B = sparse.load_npz(network_file_name)
                    # set seed
                    set_seed(seed)
                    param_set.step_size_a_w = i
                    param_set.step_size_a_pi = j
                    current_auc = run_random_SGRLD(B, args, param_set, test_pairs, train_pairs, valid_pairs,
                                                    model, 0, outfile_name, verbose=0)
                    all_auc.append(current_auc)
                    print("The AUC-ROC :", current_auc, " with step_size_a_w = ", i,
                          "and step_size_a_pi = ", j)
                    if i == 1.0 and j == 1.0:
                        auc = current_auc
                        opt_step_size_a_w = i
                        opt_step_size_a_pi = j
                    elif current_auc > auc:
                        auc = current_auc
                        opt_step_size_a_w = i
                        opt_step_size_a_pi = j
        param_set.step_size_a_w = opt_step_size_a_w
        param_set.step_size_a_pi = opt_step_size_a_pi
        args.num_burn_in = num_burn_in
        args.num_samples = num_samples
        print("The optimal initial step size are :")
        print("step_size_a_w = ", opt_step_size_a_w, "step_size_a_pi = ", opt_step_size_a_pi)

        # Save all auc for various initial step size
        outfile = outfile_name + '_all_auc.npz'
        np.savez(outfile, all_auc=all_auc)
    B = sparse.load_npz(network_file_name)
    set_seed(seed)
    auc = run_random_SGRLD(B, args, param_set, test_pairs, train_pairs, valid_pairs, model, 1, outfile_name, verbose=1)
    print("The AUC-ROC :", auc)