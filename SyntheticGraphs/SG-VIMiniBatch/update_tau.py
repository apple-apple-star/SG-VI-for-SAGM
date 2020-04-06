import tensorflow as tf
from bounds import greater_log_exp_bound

def update_tau(phi_bar_tf, tau_tf, beta_0_tf, beta_1_tf, k_tf, n_tf, curr_node_tf, global_mini_batch_indices,
               mini_batch_size_tf, step_size_RobMon_tf):

    curr_indices = tf.concat([global_mini_batch_indices, [curr_node_tf]], axis=0)

    curr_phi_bar_tf = tf.gather(phi_bar_tf, indices=curr_indices, axis=0)

    curr_phi = tf.log(tf.exp(curr_phi_bar_tf) + 1)
    curr_phi = greater_log_exp_bound(curr_phi, curr_phi_bar_tf)

    sum_phi = tf.expand_dims(tf.reduce_sum(tf.digamma(curr_phi[:, :, 0]) -
                                           tf.digamma(curr_phi[:, :, 0] + curr_phi[:, :, 1]), axis=0), axis=0)

    grad_tau = tf.squeeze(tf.stack([tf.ones([1, k_tf], tf.float64) * (n_tf + beta_0_tf),
                                    beta_1_tf - tf.multiply(tf.divide(n_tf, mini_batch_size_tf + 1), sum_phi)],
                                   axis=1)) - tau_tf

    step_size = step_size_RobMon_tf

    # Update tau
    updated_tau = tau_tf + tf.multiply(step_size, grad_tau)

    # Assign tau
    assign_tau_tf = tau_tf.assign(updated_tau)

    return [assign_tau_tf]

