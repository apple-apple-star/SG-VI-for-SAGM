
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
import timeit
import tensorflow as tf
from scipy import sparse

from calculate_auc_pp import calculate_auc_pp
from assign_w import assign_w

# for mini-batch
from update_lambda_bar import update_lambda_bar
from update_tau import update_tau
from update_phi_bar import update_phi_bar

# for tensorflow records
from tf_records import np_to_tfrecords, read_records, assign_record


def run_random_GRVI(B, args, param_set, test_pairs, train_pairs, valid_pairs, model, Train, saved, outfile_name, verbose=1):

    step_size_a_lambda = param_set.step_size_a_lambda
    step_size_a_phi = param_set.step_size_a_phi
    step_size_a_tau = param_set.step_size_a_tau
    step_size_b = param_set.step_size_b
    step_size_c = param_set.step_size_c

    network = args.network
    riemann = args.riemann
    elbo = args.elbo

    k = args.K
    mc = args.mc
    L = args.L
    M = args.m
    num_of_samples_in_records = args.num_of_samples_in_records
    num_max_iterations = args.num_max_iterations

    n = B.shape[1]
    scale_elbo = n*n/(2.0 * Train)

    outfile = outfile_name + '.npz'

    Boolean_test_links = np.squeeze(np.array(B[test_pairs[0, :], test_pairs[1, :]]))
    Boolean_train_links = np.squeeze(np.array(B[train_pairs[0, :], train_pairs[1, :]]))
    Boolean_valid_links = np.squeeze(np.array(B[valid_pairs[0, :], valid_pairs[1, :]]))

    num_tensorflow_records = np.ceil((num_max_iterations) / num_of_samples_in_records)

    test_and_valid_pair = np.hstack((test_pairs, valid_pairs))

    B[test_and_valid_pair[0, :], test_and_valid_pair[1, :]] = 1
    full_b = np.transpose(B) + B
    num_train_non_edge_each_node = (n-1)-np.squeeze(np.array(np.sum(full_b, axis=1)))  # 1 is for self-loop
    min_num_train_non_edge_each_node = min(num_train_non_edge_each_node)
    if verbose:
        print("Minimum number of non edges per node : {}".format(min_num_train_non_edge_each_node))

    full_b[test_and_valid_pair[0, :], test_and_valid_pair[1, :]] = 0
    full_b[test_and_valid_pair[1, :], test_and_valid_pair[0, :]] = 0
    num_train_edge_each_node = np.squeeze(np.array(np.sum(full_b, axis=1)))
    if verbose:
        print("Maximum number of edges per node : {}".format(max(num_train_edge_each_node)))

    B = sparse.csr_matrix((np.repeat(True, test_and_valid_pair.shape[1]),
                           (test_and_valid_pair[0, :], test_and_valid_pair[1, :])), shape=(n, n))


    full_b_test = np.transpose(B) + B
    full_b_test.setdiag(False)

    # Assert L is less than train non-edge for every node
    if L > min_num_train_non_edge_each_node and verbose:
        print("L should be less than train non-edge for every node")
        exit()

    # Tensorflow variables specific for mini-batch
    global_mini_batch_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32,
                                            name="global_mini_batch_indices")
    local_mini_batch_indices_m = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32,
                                             name="local_mini_batch_indices_m")
    curr_indices_train_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32,
                                             name="curr_indices_train_indices")
    true_curr_indices_train_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32,
                                                  name="true_curr_indices_train_indices")
    curr_indices_test_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32,
                                            name="curr_indices_test_indices")

    n_tf = tf.constant(n, dtype=tf.float64, name="n_tf")
    k_tf = tf.constant(k, dtype=tf.int32, name="k_tf")
    boolean_train_links_tf = tf.convert_to_tensor(Boolean_train_links, dtype=tf.bool, name="boolean_train_links_tf")
    train_pairs_tf = tf.convert_to_tensor(train_pairs, dtype=tf.int32, name="train_pairs_tf")
    boolean_test_links_tf = tf.convert_to_tensor(Boolean_test_links, dtype=tf.bool, name="boolean_test_links_tf")
    test_pairs_tf = tf.convert_to_tensor(test_pairs, dtype=tf.int32, name="test_pairs_tf")
    boolean_valid_links_tf = tf.convert_to_tensor(Boolean_valid_links, dtype=tf.bool, name="boolean_valid_links_tf")
    valid_pairs_tf = tf.convert_to_tensor(valid_pairs, dtype=tf.int32, name="valid_pairs_tf")

    L_tf = tf.constant(L, dtype=tf.int32, name="L_tf")
    m_tf = tf.constant(M, dtype=tf.int32, name="m_tf")
    num_train_edge_each_node_tf = tf.constant(num_train_edge_each_node, dtype=tf.int32, name="m_tf")

    # Tensorflow variables in param_set
    beta_0_tf = tf.constant(param_set.beta_0, dtype=tf.float64, name="alpha_shape_tf")
    beta_1_tf = tf.constant(param_set.beta_1, dtype=tf.float64, name="alpha_rate_tf")
    eta_0_by_1_tf = tf.constant([[param_set.eta_0], [param_set.eta_1]], dtype=tf.float64, name="eta_0_tf")
    pi_0_tf = tf.constant(model.pi0, dtype=tf.float64, name="pi_0_tf")
    mc_sample = tf.constant(mc, dtype=tf.int32, name="mc_sample")
    riemann_tf = tf.constant(riemann, dtype=tf.int32, name="riemann")
    elbo_tf = tf.constant(elbo, dtype=tf.int32, name="elbo")
    scale_elbo_tf = tf.constant(scale_elbo, dtype=tf.float64, name="scale_elbo")

    # Tensorflow variables in model
    iteration_tf = tf.placeholder(dtype=tf.float64, name="iteration_tf")
    tau_holder = tf.placeholder(tf.float64)
    lambda_bar_holder = tf.placeholder(tf.float64)
    phi_bar_holder = tf.placeholder(tf.float64)

    curr_node_tf = tf.placeholder(dtype=tf.int32, name="curr_node_tf")
    mini_batch_size_tf = tf.placeholder(dtype=tf.float64, name="mini_batch_size_tf")
    edge_tf = tf.placeholder(dtype=tf.int32, name="edge_tf")
    step_size_RobMon_tf = tf.placeholder(dtype=tf.float64, name="step_size_RobMon_tf")

    file_name_tf = tf.Variable('abc', dtype=tf.string)
    dataset = tf.data.TFRecordDataset([file_name_tf])
    dataset = dataset.map(read_records)
    dataset = dataset.batch(1)
    dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.repeat(1)

    iterator = dataset.make_initializable_iterator()

    edge = assign_record(iterator, global_mini_batch_indices,  m_tf, L_tf, num_train_edge_each_node_tf,
                         local_mini_batch_indices_m, curr_indices_train_indices, true_curr_indices_train_indices,
                         curr_indices_test_indices)

    tau_tf = tf.Variable(tau_holder, validate_shape=False, dtype=tf.float64)
    lambda_bar_tf = tf.Variable(lambda_bar_holder, validate_shape=False, dtype=tf.float64)
    phi_bar_tf = tf.Variable(phi_bar_holder, validate_shape=False, dtype=tf.float64)

    auc_pp_test = calculate_auc_pp(phi_bar_tf, lambda_bar_tf, tau_tf, eta_0_by_1_tf, beta_0_tf, beta_1_tf,
                                   test_pairs_tf, boolean_test_links_tf, pi_0_tf, elbo_tf, scale_elbo_tf)
    auc_pp_train = calculate_auc_pp(phi_bar_tf, lambda_bar_tf, tau_tf, eta_0_by_1_tf, beta_0_tf, beta_1_tf,
                                    train_pairs_tf, boolean_train_links_tf, pi_0_tf, elbo_tf, scale_elbo_tf)
    auc_pp_valid = calculate_auc_pp(phi_bar_tf, lambda_bar_tf, tau_tf, eta_0_by_1_tf, beta_0_tf, beta_1_tf,
                                    valid_pairs_tf, boolean_valid_links_tf, pi_0_tf, elbo_tf, scale_elbo_tf)

    find_tau = update_tau(phi_bar_tf, tau_tf, beta_0_tf, beta_1_tf, k_tf, n_tf, curr_node_tf, global_mini_batch_indices,
                          mini_batch_size_tf, step_size_RobMon_tf)

    find_phi_bar = update_phi_bar(phi_bar_tf, lambda_bar_tf, n_tf, k_tf, curr_indices_train_indices, mini_batch_size_tf,
                                  m_tf, curr_indices_test_indices, pi_0_tf, global_mini_batch_indices, curr_node_tf,
                                  local_mini_batch_indices_m, tau_tf, mc_sample, step_size_RobMon_tf, riemann_tf)

    find_lambda_bar = update_lambda_bar(phi_bar_tf, lambda_bar_tf, curr_node_tf, n_tf, k_tf,
                                        global_mini_batch_indices, pi_0_tf, eta_0_by_1_tf, mini_batch_size_tf,
                                        mc_sample, edge_tf, step_size_RobMon_tf, riemann_tf)

    final_w = assign_w(phi_bar_tf)

    # Output vectors to be saved
    TimeVector = []

    TestAUCvector = []
    TestPPvector = []
    TestELBOvector = []


    TrainAUCvector = []
    TrainPPvector = []
    TrainELBOvector = []

    ValidAUCvector = []
    ValidPPvector = []
    ValidELBOvector = []

    count_sample_auc = 0
    start = timeit.default_timer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create the tensorflow records
        record_file = '../records/' + network + '.tfrecords'
        np_to_tfrecords(full_b, full_b_test, num_of_samples_in_records, L, M, record_file, verbose=False)
        # Initialize the variables of the model
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict={tau_holder: model.tau, lambda_bar_holder: model.lambda_bar,
                                  phi_bar_holder: model.phi_bar})
        # Initialize the record
        record_number = 0
        sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})

        for iteration in range(num_max_iterations):
            # MINI-BATCH SAMPLING
            curr, _, _, _, _, _ = sess.run(edge)
            curr_node = curr[0, 0]
            sample_edge = curr[0, 1] == 1
            step_size_RobMon_lambda = step_size_a_lambda * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            step_size_RobMon_phi = step_size_a_phi * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            step_size_RobMon_tau = step_size_a_tau * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            if sample_edge:
                # print("For edges")
                sess.run(find_phi_bar, feed_dict={iteration_tf: iteration + 1, curr_node_tf: curr_node,
                                                  mini_batch_size_tf: num_train_edge_each_node[curr_node],
                                                  step_size_RobMon_tf: step_size_RobMon_phi})

                sess.run(find_lambda_bar, feed_dict={iteration_tf: iteration + 1, curr_node_tf: curr_node,
                                                     mini_batch_size_tf: num_train_edge_each_node[curr_node],
                                                     edge_tf: sample_edge,step_size_RobMon_tf: step_size_RobMon_lambda})

                sess.run(find_tau, feed_dict={iteration_tf: iteration + 1, curr_node_tf: curr_node,
                                              mini_batch_size_tf: num_train_edge_each_node[curr_node],
                                              step_size_RobMon_tf: step_size_RobMon_tau})


            else:
                # print("For non-edges")
                sess.run(find_phi_bar, feed_dict={iteration_tf: iteration + 1, curr_node_tf: curr_node,
                                                  mini_batch_size_tf: L,
                                                  step_size_RobMon_tf: step_size_RobMon_phi})

                sess.run(find_lambda_bar, feed_dict={iteration_tf: iteration + 1, curr_node_tf: curr_node,
                                                     mini_batch_size_tf: L, edge_tf: sample_edge,
                                                     step_size_RobMon_tf: step_size_RobMon_lambda})

                sess.run(find_tau, feed_dict={iteration_tf: iteration + 1, curr_node_tf: curr_node,
                                              mini_batch_size_tf: L,
                                              step_size_RobMon_tf: step_size_RobMon_tau})

            # COMPUTING AUC
            if (iteration + 1) % num_of_samples_in_records == 0:
                count_sample_auc = count_sample_auc + 1

                result_test = sess.run(auc_pp_test)
                TestAUCvector.append(result_test[0])
                TestPPvector.append(result_test[1])
                TestELBOvector.append(result_test[2])
                if verbose:
                    print("The current iteration is " + str(iteration + 1))
                    print("Printing results for Test set")
                    print("AUC-ROC : " + str(result_test[0]))
                    print("Perplexity : " + str(result_test[1]))
                    print("Estimated ELBO : " + str(result_test[2]))
                result_train = sess.run(auc_pp_train)
                TrainAUCvector.append(result_train[0])
                TrainPPvector.append(result_train[1])
                TrainELBOvector.append(result_train[2])
                if verbose:
                    print("Printing results for Train set")
                    print("AUC-ROC : " + str(result_train[0]))
                    print("Perplexity : " + str(result_train[1]))
                    print("Estimated ELBO : " + str(result_train[2]))
                result_valid = sess.run(auc_pp_valid)
                ValidAUCvector.append(result_valid[0])
                ValidPPvector.append(result_valid[1])
                ValidELBOvector.append(result_valid[2])
                if verbose:
                    print("Printing results for Validation set")
                    print("AUC-ROC : " + str(result_valid[0]))
                    print("Perplexity : " + str(result_valid[1]))
                    print("Estimated ELBO : " + str(result_valid[2]))
                record_number = record_number + 1
                stop_iter = timeit.default_timer()
                TimeVector.append(stop_iter - start)
                if num_tensorflow_records > record_number:
                    # create the record
                    record_file = '../records/' + network + '.tfrecords'
                    np_to_tfrecords(full_b, full_b_test, num_of_samples_in_records, L, M, record_file,
                                    verbose=False)
                    # initilize the record
                    sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})

        w = sess.run(final_w)
        stop = timeit.default_timer()
        if verbose:
            print('Time: ', stop - start)


        if saved:
            np.savez(outfile, TestAUCvector=TestAUCvector, TestPPvector=TestPPvector, TrainAUCvector=TrainAUCvector,
                     TrainPPvector=TrainPPvector, TimeVector=TimeVector, TestELBOvector=TestELBOvector,
                     TrainELBOvector=TrainELBOvector, w=w, lr_phi=param_set.step_size_a_phi,
                     lr_lambda=param_set.step_size_a_lambda,lr_tau=param_set.step_size_a_tau)

    return result_valid[0]

