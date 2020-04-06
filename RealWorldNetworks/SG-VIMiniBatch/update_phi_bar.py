import tensorflow as tf
from bounds import probability_bound, exponentional_bound, greater_log_exp_bound, replace_zero
import tensorflow_probability as tfp

tfd = tfp.distributions


def update_phi_bar(phi_bar_tf, lambda_bar_tf, n_tf, k_tf, curr_indices_train_indices, mini_batch_size_tf,
                   m_tf, curr_indices_test_indices, pi_0_tf, global_mini_batch_indices,
                   curr_node_tf, local_mini_batch_indices_m, tau_tf, mc_sample,
                   step_size_RobMon_tf, riemann_tf):
    float_one = tf.constant(1.0, dtype=tf.float64)
    float_two = tf.constant(2.0, dtype=tf.float64)

    # Sampling pi
    lambd = tf.log(tf.exp(lambda_bar_tf) + 1)  # 2 * K
    lambd = greater_log_exp_bound(lambd, lambda_bar_tf)  # 2 * K


    pi_dist = tfd.Beta(lambd[0, :], lambd[1, :])
    pi = pi_dist.sample([mc_sample])  # mc_sample, K
    pi = probability_bound(pi, tf.ones((mc_sample, k_tf), tf.float64))

    # Find the nodes in mini-batch node pairs
    curr_indices = tf.concat([global_mini_batch_indices, [curr_node_tf]], axis=0)

    # Take out the phi_bar_tf for the nodes Mini-batch node pairs
    curr_phi_bar_tf = tf.gather(phi_bar_tf, indices=curr_indices, axis=0)  # Mini_batch_size * K * 2
    curr_phi = tf.log(tf.exp(curr_phi_bar_tf) + 1)
    curr_phi = greater_log_exp_bound(curr_phi, curr_phi_bar_tf)  # Mini_batch_size * K * 2

    curr_w_bar_dist = tfd.Gamma(curr_phi, float_one)
    curr_w_bar = curr_w_bar_dist.sample([mc_sample])  # mc_sample * Mini_batch_size * K * 2
    curr_w = curr_w_bar[:, :, :, 0]/(curr_w_bar[:, :, :, 0] + curr_w_bar[:, :, :, 1])  # mc_sample * Mini_batch_size * K
    curr_w = probability_bound(curr_w, tf.ones((mc_sample, tf.cast(mini_batch_size_tf, dtype=tf.int32) + 1, k_tf),
                                               tf.float64))  # mc_sample * Mini_batch_size * K

    # Common for g^rep, g^corr and entropy
    psi_1_term = tf.polygamma(float_one, curr_phi)  # Mini_batch_size * K * 2

    ## Computing the entropy
    common_term_entropy = tf.multiply(curr_phi[:, :, 0] + curr_phi[:, :, 1] - 2,
                                      tf.polygamma(float_one, curr_phi[:, :, 0] + curr_phi[:, :, 1]))  # Mini_batch_size * K
    entropy_phi = tf.expand_dims(common_term_entropy, axis=2) - tf.multiply(curr_phi - 1, psi_1_term)  # Mini_batch_size * K * 2


    # Common for g^rep, g^corr
    log_psi_term = tf.log(curr_w_bar) - tf.digamma(tf.expand_dims(curr_phi, axis=0))  # mc_sample * Mini_batch_size * K * 2
    psi_2_by_psi_1 = tf.divide(tf.polygamma(float_two, curr_phi), 2*psi_1_term)  # Mini_batch_size * K * 2
    common_grep_gcorr = log_psi_term * psi_2_by_psi_1 + psi_1_term  # mc_sample * Mini_batch_size * K * 2
    tau_divide = tf.divide(tau_tf[0, :], tau_tf[1, :])-1 # K,

    grep_w_term = tf.multiply(curr_w, 1-curr_w)  # mc_sample * Mini_batch_size * K

    grep_without_f = tf.multiply(tf.stack([grep_w_term,  -grep_w_term], axis=3), common_grep_gcorr)  # mc_sample * Mini_batch_size * K * 2
    gcorr_without_f = tf.multiply((tf.expand_dims(curr_phi, axis=0)-curr_w_bar),
                                  common_grep_gcorr) + log_psi_term + tf.expand_dims(psi_2_by_psi_1, axis=0)  # mc_sample * Mini_batch_size * K * 2

    # gather the local mini-batch
    scale = tf.divide(n_tf, tf.cast(m_tf, dtype=tf.float64))

    local_mini_batch_curr_phi_bar = tf.gather(phi_bar_tf, local_mini_batch_indices_m, axis=0)  # Mini_batch_size * m * K * 2
    local_mini_batch_curr_phi = tf.log(tf.exp(local_mini_batch_curr_phi_bar) + 1)
    local_mini_batch_curr_phi = greater_log_exp_bound(local_mini_batch_curr_phi, local_mini_batch_curr_phi_bar)

    local_mini_batch_curr_w_dist = tfd.Beta(local_mini_batch_curr_phi[:, :, :, 0], local_mini_batch_curr_phi[:, :, :, 1])
    local_mini_batch_curr_w = local_mini_batch_curr_w_dist.sample([mc_sample])   # mc_sample * Mini_batch_size * m * K
    local_mini_batch_curr_w = probability_bound(local_mini_batch_curr_w,
                                                tf.ones((mc_sample, tf.cast(mini_batch_size_tf, dtype=tf.int32) + 1,
                                                         m_tf, k_tf), tf.float64))  # mc_sample * Mini_batch_size * m * K


    w_pi_reduced_m = tf.multiply(local_mini_batch_curr_w,
                                 tf.expand_dims(tf.expand_dims(pi, axis=1), axis=1))  # mc_sample * Mini_batch_size * m * K

    ww_pi = tf.multiply(tf.expand_dims(curr_w, axis=2), w_pi_reduced_m)  # mc_sample * Mini_batch_size * m * K

    log_sum_ww_pi = tf.log(1 - pi_0_tf) + tf.reduce_sum(tf.log(1 - ww_pi), axis=3)  # mc_sample * Mini_batch_size * m

    test_diag = tf.sparse_to_dense(sparse_indices=curr_indices_test_indices, sparse_values=float_one,
                                   output_shape=[tf.cast(mini_batch_size_tf, dtype=tf.int32) + 1, m_tf])  # Mini_batch_size * m

    train_edge = tf.sparse_to_dense(sparse_indices=curr_indices_train_indices, sparse_values=float_one,
                                    output_shape=[tf.cast(mini_batch_size_tf, dtype=tf.int32) + 1, m_tf])  # Mini_batch_size * m


    train_edge_log_prob = tf.expand_dims(train_edge, axis=0) * log_sum_ww_pi  # type: Matrix (mc_sample * Mini_batch_size * m)

    train_edges_log_probs_non_edge_minus_edge = train_edge_log_prob - \
                                                tf.log(1 - tf.exp(train_edge_log_prob))  # type: Matrix (mc_sample * Mini_batch_size * m)

    train_edges_log_probs_non_edge_minus_edge = replace_zero(train_edges_log_probs_non_edge_minus_edge, train_edge_log_prob)


    # Computing f
    log_probability = log_sum_ww_pi - tf.expand_dims(test_diag, axis=0) * log_sum_ww_pi - \
                      train_edges_log_probs_non_edge_minus_edge  # mc_sample * Mini_batch_size * m

    f = tf.multiply(scale, tf.expand_dims(tf.reduce_sum(log_probability, axis=2), axis=2)) + \
        tf.digamma(tau_tf[0, :]) - tf.log(tau_tf[1, :]) + tf.multiply(tf.expand_dims
                                                                      (tf.expand_dims(tau_divide, axis=0), axis=0),
                                                                      tf.log(curr_w))  # mc_sample * Mini_batch_size * K


    gcorr = tf.reduce_sum(tf.expand_dims(f, axis=3)*gcorr_without_f, axis=0)  # Mini_batch_size * K * 2

    # Computing derivative of f

    w_mask = 1 - (2.0*train_edge + test_diag)  # Mini_batch_size * L

    h = tf.multiply(tf.expand_dims(w_mask, axis=0), tf.exp(train_edges_log_probs_non_edge_minus_edge))  # mc_sample * Mini_batch_size * m

    # Common Term
    common_term_reduced_m = tf.divide(-w_pi_reduced_m, 1 - ww_pi)  # mc_sample * Mini_batch_size * m * K
    row_sum_reduced_m = tf.reduce_sum(tf.multiply(common_term_reduced_m, tf.expand_dims(h, axis=3)),
                                      axis=2)  # mc_sample * Mini_batch_size * K

    derivatives_f = tf.multiply(scale, row_sum_reduced_m) + tf.divide(tf.expand_dims(tf.expand_dims(tau_divide, axis=0),
                                                                                     axis=0), curr_w)  # mc_sample * Mini_batch_size * K

    grep = tf.reduce_sum(tf.expand_dims(derivatives_f, axis=3) * grep_without_f, axis=0)  # Mini_batch_size * K * 2

    # Gradient of phi
    def riemann_grad():
        return tf.multiply(curr_phi, grep + gcorr + entropy_phi)
    def euclid_grad():
        return grep + gcorr + entropy_phi
    grad_phi = tf.cond(tf.equal(riemann_tf, 1), riemann_grad, euclid_grad)


    # Gradient of phi bar
    grad_phi_bar = tf.divide(grad_phi, 1 + tf.exp(- curr_phi_bar_tf))
    step_size = step_size_RobMon_tf

    # Update phi_bar
    updated_phi_bar = curr_phi_bar_tf + tf.multiply(step_size, grad_phi_bar)
    updated_phi_bar = exponentional_bound(updated_phi_bar, tf.ones((tf.cast(mini_batch_size_tf, dtype=tf.int32) + 1,
                                                                    k_tf, 2), tf.float64))

    # Assign phi_bar
    assign_phi_bar_tf = tf.scatter_update(phi_bar_tf, indices=curr_indices, updates=updated_phi_bar)

    return [assign_phi_bar_tf]