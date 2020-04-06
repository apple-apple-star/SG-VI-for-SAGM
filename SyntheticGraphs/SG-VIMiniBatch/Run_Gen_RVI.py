
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from scipy import sparse
from Gen_RVI_iteration import run_random_GRVI
from util import set_seed
from create_mask_network_sparse_equal import create_mask_network_sparse_equal
from sample_initial_parameters import sample_initial_parameters_model



def run_GRVI(args, param_set):
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

    outfile_name = args.outfile + network + 'seed' + str(seed) + 'lr_lambda' + \
                   str(param_set.step_size_a_lambda) + 'lr_phi' + str(param_set.step_size_a_phi) + 'lr_tau' + \
                   str(param_set.step_size_a_tau) + 'riemann' + str(args.riemann) + 'elbo' + str(args.elbo) + \
                   'mc' + str(args.mc) + 'L' + str(args.L) + 'M' + str(args.m) + 'K' + str(k) + 'max iterations' + \
                   str(args.num_max_iterations) + 'test_ratio' + str(args.Test_ratio)

    model = sample_initial_parameters_model(k, n)
    if param_set.step_size_a_lambda == 0 or param_set.step_size_a_phi == 0 or param_set.step_size_a_tau ==0:
        outfile_name = args.outfile + 'LineSearch_' + network + 'seed' + str(seed) + 'lr_lambda' + \
                       str(param_set.step_size_a_lambda) + 'lr_phi' + str(param_set.step_size_a_phi) + 'lr_tau' + \
                       str(param_set.step_size_a_tau) + 'riemann' + str(args.riemann) + 'elbo' + str(args.elbo) + \
                       'mc' + str(args.mc) + 'L' + str(args.L) + 'M' + str(args.m) + 'K' + str(k) + 'max iterations' + \
                       str(args.num_max_iterations) + 'test_ratio' + str(args.Test_ratio)

        all_auc = []
        max_iter = args.num_max_iterations
        args.num_max_iterations = 2000

        for i in initial_step_size_range:  # lambda
            for j in initial_step_size_range:  # phi
                for l in initial_step_size_range:  # tau
                    B = sparse.load_npz(network_file_name)
                    # set seed
                    set_seed(seed)
                    param_set.step_size_a_lambda = i
                    param_set.step_size_a_phi = j
                    param_set.step_size_a_tau = l
                    current_auc = run_random_GRVI(B, args, param_set, test_pairs, train_pairs, valid_pairs,
                                                   model, Train, 0, outfile_name, verbose=0)
                    all_auc.append(current_auc)
                    print("The AUC-ROC :", current_auc, " step_size_a_lambda= ", i,
                          "step_size_a_phi = ", j, " and step_size_a_tau = ", l)
                    if i==1.0 and j==1.0 and l==1.0:
                        auc = current_auc
                        opt_step_size_a_lambda = i
                        opt_step_size_a_phi = j
                        opt_step_size_a_tau = l
                    elif current_auc > auc:
                        auc = current_auc
                        opt_step_size_a_lambda = i
                        opt_step_size_a_phi = j
                        opt_step_size_a_tau = l
        param_set.step_size_a_lambda = opt_step_size_a_lambda
        param_set.step_size_a_phi = opt_step_size_a_phi
        param_set.step_size_a_tau = opt_step_size_a_tau
        args.num_max_iterations = max_iter
        print("The optimal initial step size are :")
        print("step_size_a_lambda = ", opt_step_size_a_lambda, "step_size_a_phi = ", opt_step_size_a_phi,
              "step_size_a_tau = ", opt_step_size_a_tau)

        # Save all auc for various initial step size
        outfile = outfile_name + '_all_auc.npz'
        np.savez(outfile, all_auc=all_auc)
    B = sparse.load_npz(network_file_name)
    set_seed(seed)
    auc = run_random_GRVI(B, args, param_set, test_pairs, train_pairs, valid_pairs,
                           model, Train, 1, outfile_name, verbose=1)
    print("The AUC-ROC :", auc)