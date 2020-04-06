import tensorflow as tf
from bounds import probability_bound, exponentional_bound, greater_log_exp_bound
import tensorflow_probability as tfp
tfd = tfp.distributions

def update_lambda_bar(phi_bar_tf, lambda_bar_tf, curr_node_tf, n_tf, k_tf, global_mini_batch_indices, pi_0_tf,
                      eta_0_by_1_tf, mini_batch_size_tf, mc_sample, edge_tf, step_size_RobMon_tf, riemann_tf):

    float_one = tf.constant(1.0, dtype=tf.float64)
    float_two = tf.constant(2.0, dtype=tf.float64)

    lambd = tf.log(tf.exp(lambda_bar_tf) + 1)
    lambd = greater_log_exp_bound(lambd, lambda_bar_tf)  # 2 * K

    # Sampling pi
    pi_bar_dist = tfd.Gamma(lambd, float_one)
    pi_bar = pi_bar_dist.sample([mc_sample])  # mc_sample * 2 * K
    pi = pi_bar[:, 0, :] / (pi_bar[:, 0, :] + pi_bar[:, 1, :])  # mc_sample * K
    pi = probability_bound(pi, tf.ones((mc_sample, k_tf), tf.float64))  # mc_sample * K

    # Common for g^rep, g^corr and entropy
    psi_1_term = tf.polygamma(float_one, lambd)  # 2 * K

    ## Computing the entropy
    common_term_entropy = tf.multiply(lambd[0, :] + lambd[1, :] - 2,
                                      tf.polygamma(float_one, lambd[0, :] + lambd[1, :]))  # K,
    entropy_lambd = tf.expand_dims(common_term_entropy, 0) - tf.multiply(lambd - 1, psi_1_term)  # 2 * K

    # Common for g^rep, g^corr
    log_psi_term = tf.log(pi_bar) - tf.digamma(tf.expand_dims(lambd, axis=0))  # mc_sample * 2 * K
    psi_2_by_psi_1 = tf.divide(tf.polygamma(float_two, lambd), 2 * psi_1_term)  # 2 * K
    common_grep_gcorr = log_psi_term * tf.expand_dims(psi_2_by_psi_1, axis=0) + tf.expand_dims(psi_1_term, axis=0)  # mc_sample * 2 * K
    grep_pi_term = tf.multiply(pi, 1 - pi)  # mc_sample * K

    grep_without_f = tf.multiply(tf.stack([grep_pi_term, -grep_pi_term], axis=1), common_grep_gcorr)  # mc_sample * 2 * K
    gcorr_without_f = tf.multiply((lambd - pi_bar), common_grep_gcorr) + log_psi_term + psi_2_by_psi_1  # mc_sample * 2 * K

    # gather the local mini-batch
    scale = tf.cond(tf.equal(edge_tf, 1), lambda: n_tf,
                    lambda: tf.divide(tf.multiply(n_tf, n_tf), mini_batch_size_tf))

    # curr_node
    phi_bar_curr_node = tf.gather(phi_bar_tf, indices=[curr_node_tf], axis=0)  # 1 * K * 2
    phi_curr_node = tf.log(tf.exp(phi_bar_curr_node) + 1)  # 1 * K * 2
    phi_curr_node = greater_log_exp_bound(phi_curr_node, phi_bar_curr_node)  # 1 * K * 2
    w_curr_node_dist = tfd.Beta(phi_curr_node[:, :, 0], phi_curr_node[:, :, 1])
    w_curr_node = w_curr_node_dist.sample([mc_sample])  # mc_sample * 1 * K
    w_curr_node = probability_bound(w_curr_node, tf.ones((mc_sample, tf.cast(1, dtype=tf.int32), k_tf),
                                                         tf.float64))  # mc_sample * 1 * K

    # neighbour_node
    def empty():
        w_neighbour_node = tf.ones((mc_sample, tf.cast(0, dtype=tf.int32), k_tf), tf.float64)
        return w_neighbour_node

    def non_empty():
        phi_bar_neighbour_node = tf.gather(phi_bar_tf, indices=global_mini_batch_indices, axis=0)  # mini_batch_size_tf * K * 2
        phi_neighbour_node = tf.log(tf.exp(phi_bar_neighbour_node) + 1)
        phi_neighbour_node = greater_log_exp_bound(phi_neighbour_node, phi_bar_neighbour_node)  # mini_batch_size_tf * K * 2
        w_neighbour_node_dist = tfd.Beta(phi_neighbour_node[:, :, 0], phi_neighbour_node[:, :, 1])
        w_neighbour_node = w_neighbour_node_dist.sample([mc_sample])  # mc_sample * mini_batch_size_tf * K
        w_neighbour_node = probability_bound(w_neighbour_node,
                                             tf.ones((mc_sample, tf.cast(mini_batch_size_tf, dtype=tf.int32), k_tf),
                                                     tf.float64))  # mc_sample * mini_batch_size_tf * K
        return w_neighbour_node

    w_neighbour_node = tf.cond(tf.equal(mini_batch_size_tf, 0), empty, non_empty)  # mc_sample * mini_batch_size_tf * K

    ww = tf.multiply(w_curr_node, w_neighbour_node)  # mc_sample * mini_batch_size_tf * K

    wwpi = tf.multiply(ww, tf.expand_dims(pi, axis=1))  # mc_sample * mini_batch_size_tf * K

    log_term = tf.log(1 - wwpi)  # mc_sample * mini_batch_size_tf * K

    log_non_edge_prob = tf.reduce_sum(log_term, axis=2) + tf.log(1 - pi_0_tf)  # mc_sample * mini_batch_size_tf

    edge_prob = (1 - tf.exp(log_non_edge_prob))  # mc_sample * mini_batch_size_tf

    # Computing f
    def edge_log_prob():
        log_probability = tf.log(edge_prob)  # mc_sample * mini_batch_size_tf
        return log_probability

    def non_edge_log_prob():
        log_probability = log_non_edge_prob  # mc_sample * mini_batch_size_tf
        return log_probability
    log_probability = tf.cond(tf.equal(edge_tf, 1), edge_log_prob, non_edge_log_prob)

    f = tf.multiply(scale, tf.expand_dims(tf.reduce_sum(log_probability, axis=1), axis=1)) + \
        tf.lgamma(eta_0_by_1_tf[0] + eta_0_by_1_tf[1]) - tf.lgamma(eta_0_by_1_tf[0]) - tf.lgamma(eta_0_by_1_tf[1]) + \
        (eta_0_by_1_tf[0] - 1) * tf.log(pi) + (eta_0_by_1_tf[1] - 1) * tf.log(1 - pi)  # mc_sample * K

    gcorr = tf.reduce_sum(tf.expand_dims(f, axis=1) * gcorr_without_f, axis=0)  # 2 * K


    def edge_h():
        return tf.divide(tf.multiply(ww, tf.exp(tf.expand_dims(log_non_edge_prob, axis=2) - log_term)),
                         tf.expand_dims(edge_prob, axis=2))
    def non_edge_h():
        return tf.divide(-ww, 1 - wwpi)
    h_pi_k_term = tf.cond(tf.equal(edge_tf, 1), edge_h, non_edge_h)  # mc_sample * mini_batch_size_tf * K

    row_sum = tf.multiply(scale, tf.reduce_sum(h_pi_k_term, axis=1))  # mc_sample * K

    derivatives_f = row_sum + tf.divide(eta_0_by_1_tf[0] - 1, pi) - tf.divide(eta_0_by_1_tf[1] - 1,
                                                                              1 - pi)  # mc_sample * K

    grep = tf.reduce_sum(tf.expand_dims(derivatives_f, axis=1) * grep_without_f, axis=0)  # 2 * K

    # Gradient of lambda
    def riemann_grad():
        return tf.multiply(lambd, grep + gcorr + entropy_lambd)
    def euclid_grad():
        return grep + gcorr + entropy_lambd
    grad_lambd = tf.cond(tf.equal(riemann_tf, 1), riemann_grad, euclid_grad)

    # Gradient of lambda bar
    grad_lambd_bar = tf.divide(grad_lambd, 1 + tf.exp(- lambda_bar_tf))  # 2, K

    step_size = step_size_RobMon_tf

    # Update lambda_bar
    updated_lambda_bar = lambda_bar_tf + tf.multiply(step_size, grad_lambd_bar)

    updated_lambda_bar = exponentional_bound(updated_lambda_bar, tf.ones((2, k_tf), tf.float64))

    # Assign lambda_bar
    assign_lambda_bar_tf = lambda_bar_tf.assign(updated_lambda_bar)

    return [assign_lambda_bar_tf]
