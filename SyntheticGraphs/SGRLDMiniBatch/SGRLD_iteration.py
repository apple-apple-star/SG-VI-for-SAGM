from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
import timeit
import tensorflow as tf
from scipy import sparse

from sample_alpha_gibbs_sagm import sample_alpha_gibbs_sagm
from calculate_auc_pp import calculate_auc_pp, calculate_auc_pp_collections, calculate_avg_auc

# for tensorflow records
from tf_records import np_to_tfrecords, read_records, assign_record

# for mini-batch
from compute_mini_gradient_pi import compute_mini_gradient_pi
from compute_mini_gradient_w import compute_mini_gradient_w

def run_random_SGRLD(B, args, param_set, test_pairs, train_pairs, valid_pairs, model, saved, outfile_name, verbose=1):

    step_size_a_w = param_set.step_size_a_w
    step_size_a_pi = param_set.step_size_a_pi
    step_size_b = param_set.step_size_b
    step_size_c = param_set.step_size_c

    network = args.network

    k = args.K
    L = args.L
    M = args.m

    num_of_samples_in_records = args.num_of_samples_in_records
    num_burn_in = args.num_burn_in
    num_samples = args.num_samples

    n = B.shape[1]

    # step_size_a_pi = args.lrpi
    outfile = outfile_name + '.npz'
    if verbose:
        print(outfile)


    Boolean_test_links = np.squeeze(np.array(B[test_pairs[0, :], test_pairs[1, :]]))
    Boolean_train_links = np.squeeze(np.array(B[train_pairs[0, :], train_pairs[1, :]]))
    Boolean_valid_links = np.squeeze(np.array(B[valid_pairs[0, :], valid_pairs[1, :]]))

    num_tensorflow_records = np.ceil((num_burn_in + num_samples) / num_of_samples_in_records)

    test_and_valid_pair = np.hstack((test_pairs, valid_pairs))

    B[test_and_valid_pair[0, :], test_and_valid_pair[1, :]] = 1
    full_b = np.transpose(B) + B
    num_train_non_edge_each_node = (n-1)-np.squeeze(np.array(np.sum(full_b, axis=1)))  # 1 is for self-loop
    min_num_train_non_edge_each_node = min(num_train_non_edge_each_node)
    if verbose:
        print("Minimum number of non edges per node : {}".format(min_num_train_non_edge_each_node))
    B[test_and_valid_pair[0, :], test_and_valid_pair[1, :]] = 0
    full_b = np.transpose(B) + B
    num_train_edge_each_node = np.squeeze(np.array(np.sum(full_b, axis=1)))

    total_test_edge_prob = np.zeros(test_pairs.shape[1])
    if verbose:
        print("Maximum number of edges per node : {}".format(max(num_train_edge_each_node)))


    B = sparse.csr_matrix((np.repeat(True, test_and_valid_pair.shape[1]),
                           (test_and_valid_pair[0, :], test_and_valid_pair[1, :])), shape=(n, n))
    full_b_test = np.transpose(B) + B
    full_b_test.setdiag(True)

    # Assert L is less than train non-edge for every node
    if L > min_num_train_non_edge_each_node and verbose:
        print("L should be less than train non-edge for every node")
        exit()

    # Tensorflow variables
    w_holder = tf.placeholder(tf.float32)
    sum_w_holder = tf.placeholder(tf.float32)
    w_tf = tf.Variable(w_holder, validate_shape=False, dtype=tf.float32)
    sum_w_bar_tf = tf.Variable(sum_w_holder, validate_shape=False, dtype=tf.float32)

    pi_tf = tf.Variable(model.pi, dtype=tf.float32, name="pi_tf")
    sum_pi_bar_tf = tf.Variable(model.sum_pi_bar, dtype=tf.float32, name="sum_pi_bar_tf")
    alpha_tf = tf.Variable(model.alpha, dtype=tf.float32, name="alpha_tf")

    test_pairs_tf = tf.convert_to_tensor(test_pairs, dtype=tf.int32, name="test_pairs_tf")
    boolean_test_links_tf = tf.convert_to_tensor(Boolean_test_links, dtype=tf.bool, name="boolean_test_links_tf")
    train_pairs_tf = tf.convert_to_tensor(train_pairs, dtype=tf.int32, name="train_pairs_tf")
    boolean_train_links_tf = tf.convert_to_tensor(Boolean_train_links, dtype=tf.bool, name="boolean_train_links_tf")
    valid_pairs_tf = tf.convert_to_tensor(valid_pairs, dtype=tf.int32, name="valid_pairs_tf")
    boolean_valid_links_tf = tf.convert_to_tensor(Boolean_valid_links, dtype=tf.bool, name="boolean_valid_links_tf")

    beta_0_tf = tf.constant(1., dtype=tf.float32, name="alpha_shape_tf")
    beta_1_tf = tf.constant(1., dtype=tf.float32, name="alpha_rate_tf")
    n_tf = tf.constant(n, dtype=tf.float32, name="n_tf")
    k_tf = tf.constant(k, dtype=tf.int32, name="k_tf")
    eta_0_by_1_tf = tf.constant([[param_set.eta_0], [param_set.eta_1]], dtype=tf.float32, name="eta_0_tf")
    pi_0_tf = tf.constant(0.00005, dtype=tf.float32, name="pi_0_tf")

    step_size_w_tf = tf.placeholder(dtype=tf.float32, name="step_size_w_tf")
    step_size_pi_tf = tf.placeholder(dtype=tf.float32, name="step_size_pi_tf")

    # Tensorflow variables specific for mini-batch
    global_mini_batch_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="global_mini_batch_indices")
    local_mini_batch_indices_m = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="local_mini_batch_indices_m")
    curr_indices_train_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="curr_indices_train_indices")
    true_curr_indices_train_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="true_curr_indices_train_indices")
    curr_indices_test_indices = tf.Variable([[1, 1]], validate_shape=False, dtype=tf.int32, name="curr_indices_test_indices")
    total_test_edge_prob_tf = tf.Variable(total_test_edge_prob, total_test_edge_prob.shape, dtype=tf.float32, name="total_test_edge_prob_tf")
    file_name_tf = tf.Variable('abc', dtype=tf.string)

    L_tf = tf.constant(L, dtype=tf.int32, name="L_tf")
    m_tf = tf.constant(M, dtype=tf.int32, name="m_tf")
    num_train_edge_each_node_tf = tf.constant(num_train_edge_each_node, dtype=tf.int32, name="m_tf")

    curr_node_tf = tf.placeholder(dtype=tf.int32, name="curr_node_tf")
    mini_batch_size_tf = tf.placeholder(dtype=tf.int32, name="mini_batch_size_tf")
    count_tf = tf.placeholder(dtype=tf.int32, name="count_tf")
    edge_tf = tf.placeholder(dtype=tf.int32, name="edge_tf")


    # Tensorflow Computational Graph
    auc_pp_test = calculate_auc_pp(w_tf, test_pairs_tf, boolean_test_links_tf, pi_0_tf, pi_tf)
    auc_pp_test_collections = calculate_auc_pp_collections(w_tf, test_pairs_tf, boolean_test_links_tf, pi_0_tf, pi_tf, total_test_edge_prob_tf)
    avg_auc_test = calculate_avg_auc(total_test_edge_prob_tf, boolean_test_links_tf, count_tf)
    auc_pp_train = calculate_auc_pp(w_tf, train_pairs_tf, boolean_train_links_tf, pi_0_tf, pi_tf)
    auc_pp_valid = calculate_auc_pp(w_tf, valid_pairs_tf, boolean_valid_links_tf, pi_0_tf, pi_tf)

    find_mini_gradient_pi = compute_mini_gradient_pi(pi_tf, sum_pi_bar_tf, pi_0_tf, w_tf, curr_node_tf, n_tf, k_tf,
                                                     step_size_pi_tf, eta_0_by_1_tf, global_mini_batch_indices, edge_tf,
                                                     mini_batch_size_tf)


    find_mini_gradient_w = compute_mini_gradient_w(w_tf, sum_w_bar_tf, pi_tf, pi_0_tf, alpha_tf, curr_node_tf, n_tf,
                                                   k_tf, step_size_w_tf, mini_batch_size_tf, m_tf,
                                                   local_mini_batch_indices_m, global_mini_batch_indices,
                                                   curr_indices_train_indices, true_curr_indices_train_indices,
                                                   curr_indices_test_indices)

    find_alpha = sample_alpha_gibbs_sagm(alpha_tf, beta_0_tf, beta_1_tf, w_tf, n_tf)

    dataset = tf.data.TFRecordDataset([file_name_tf])
    dataset = dataset.map(read_records)
    dataset = dataset.batch(1)
    dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.repeat(1)
    iterator = dataset.make_initializable_iterator()
    edge = assign_record(iterator, global_mini_batch_indices,  m_tf, L_tf, num_train_edge_each_node_tf,
                                        local_mini_batch_indices_m, curr_indices_train_indices,
                                        true_curr_indices_train_indices, curr_indices_test_indices)



    # Output vectors to be saved
    TestAUCvector = []
    TrainAUCvector = []
    ValidAUCvector = []
    TestPPvector = []
    TrainPPvector = []
    ValidPPvector = []
    TimeVector = []
    total_w = np.zeros((n, k))

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
        sess.run(init, feed_dict={w_holder: model.w, sum_w_holder: model.sum_w_bar})

        # Initialize the record
        record_number = 0
        sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})

        if verbose:
            print("Starting the burn_in period")
        for iteration in range(num_burn_in):
            # print("Current iteration {}".format(iteration))
            step_size_w = step_size_a_w * pow((1 + float(iteration) / param_set.step_size_b), (-step_size_c))
            step_size_pi = step_size_a_pi * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            # MINI-BATCH SAMPLING
            curr, _, _, _, _, _ = sess.run(edge)
            curr_node = curr[0, 0]
            sample_edge = curr[0, 1] == 1
            if sample_edge:
                # Sample w and pi if train edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: num_train_edge_each_node[curr_node]})
                sess.run(find_mini_gradient_pi, feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi,
                                                           edge_tf: sample_edge, mini_batch_size_tf:
                                                               num_train_edge_each_node[curr_node]})

            else:
                # Sample w and pi if train non-edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: L})
                sess.run(find_mini_gradient_pi, feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi,
                                                           edge_tf: sample_edge, mini_batch_size_tf: L})

            # SAMPLING ALPHA
            sess.run(find_alpha)
            # COMPUTING AUC
            if (iteration+1) % num_of_samples_in_records == 0:
                if verbose:
                    print("The current iteration is " + str(iteration+1))
                # print(pi)
                result_test = sess.run(auc_pp_test)
                TestAUCvector.append(result_test[0])
                TestPPvector.append(result_test[1])
                if verbose:
                    print("Printing results for Test set")
                    print("auc tf : " + str(result_test[0]))
                    print("pp tf : " + str(result_test[1]))
                result_train = sess.run(auc_pp_train)
                TrainAUCvector.append(result_train[0])
                TrainPPvector.append(result_train[1])
                if verbose:
                    print("Printing results for Train set")
                    print("auc tf : " + str(result_train[0]))
                    print("pp tf : " + str(result_train[1]))
                result_valid = sess.run(auc_pp_valid)
                ValidAUCvector.append(result_valid[0])
                ValidPPvector.append(result_valid[1])
                if verbose:
                    print("Printing results for Validation set")
                    print("auc tf : " + str(result_valid[0]))
                    print("pp tf : " + str(result_valid[1]))
                record_number = record_number + 1
                stop_iter = timeit.default_timer()
                TimeVector.append(stop_iter-start)
                if num_tensorflow_records > record_number:
                    # create the record
                    record_file = '../records/' + network + '.tfrecords'
                    np_to_tfrecords(full_b, full_b_test, num_of_samples_in_records, L, M, record_file, verbose=False)
                    # initilize the record
                    sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})

        if verbose:
            print("Starting collecting samples")
        for iteration in range(num_burn_in, num_burn_in + num_samples):
            # print("Current iteration {}".format(iteration))
            step_size_w = step_size_a_w * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            step_size_pi = step_size_a_pi * pow((1 + float(iteration) / step_size_b), (-step_size_c))
            # MINI-BATCH SAMPLING
            curr, _, _, _, _, _ = sess.run(edge)
            curr_node = curr[0, 0]
            sample_edge = curr[0, 1] == 1
            if sample_edge:
                # Sample w and pi if train edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: num_train_edge_each_node[curr_node]})
                sess.run(find_mini_gradient_pi, feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi,
                                                           edge_tf: sample_edge, mini_batch_size_tf:
                                                              num_train_edge_each_node[curr_node]})

            else:
                # Sample w and pi if train non-edges of curr_node
                sess.run(find_mini_gradient_w, feed_dict={curr_node_tf: curr_node, step_size_w_tf: step_size_w,
                                                          mini_batch_size_tf: L})
                sess.run(find_mini_gradient_pi, feed_dict={curr_node_tf: curr_node, step_size_pi_tf: step_size_pi,
                                                           edge_tf: sample_edge, mini_batch_size_tf: L})
            total_w = total_w + w_tf.eval()

            # SAMPLING ALPHA
            sess.run(find_alpha)
            # COMPUTING AUC
            if (iteration + 1) % num_of_samples_in_records == 0:
                count_sample_auc = count_sample_auc + 1
                if verbose:
                    print("The current iteration is " + str(iteration + 1))
                result_test = sess.run(auc_pp_test_collections)
                TestAUCvector.append(result_test[0])
                TestPPvector.append(result_test[1])
                if verbose:
                    print("Printing results for Test set")
                    print("auc tf : " + str(result_test[0]))
                    print("pp tf : " + str(result_test[1]))
                result_train = sess.run(auc_pp_train)
                TrainAUCvector.append(result_train[0])
                TrainPPvector.append(result_train[1])
                if verbose:
                    print("Printing results for Train set")
                    print("auc tf : " + str(result_train[0]))
                    print("pp tf : " + str(result_train[1]))
                result_valid = sess.run(auc_pp_valid)
                ValidAUCvector.append(result_valid[0])
                ValidPPvector.append(result_valid[1])
                if verbose:
                    print("Printing results for Validation set")
                    print("auc tf : " + str(result_valid[0]))
                    print("pp tf : " + str(result_valid[1]))
                record_number = record_number + 1
                stop_iter = timeit.default_timer()
                TimeVector.append(stop_iter - start)
                if num_tensorflow_records > record_number:
                    # create the record
                    record_file = '../records/' + network + '.tfrecords'
                    np_to_tfrecords(full_b, full_b_test, num_of_samples_in_records, L, M, record_file, verbose=False)
                    # initilize the record
                    sess.run(iterator.initializer, feed_dict={file_name_tf: record_file})

        avg_auc_test = sess.run(avg_auc_test, feed_dict={count_tf: count_sample_auc})

    stop = timeit.default_timer()
    if verbose:
        print('Time: ', stop - start)
        print('Avg auc : ', avg_auc_test)
    if saved:
        np.savez(outfile, AvgAUC=avg_auc_test, TestAUCvector=TestAUCvector, TestPPvector=TestPPvector,
                 TrainAUCvector=TrainAUCvector, TrainPPvector=TrainPPvector, TimeVector=TimeVector, w=total_w/num_samples)

    return result_valid[0]
