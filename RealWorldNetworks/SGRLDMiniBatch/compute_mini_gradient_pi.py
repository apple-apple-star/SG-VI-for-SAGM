import tensorflow as tf

def compute_mini_gradient_pi(pi_tf, sum_pi_bar_tf, pi_0_tf, w_tf, curr_node_tf, n_tf, k_tf, step_size_pi_tf,
                             eta_0_by_1_tf, global_mini_batch_indices, edge_tf, mini_batch_size_tf):

    #  Calculating H
    ww = tf.multiply(tf.gather(w_tf, indices=curr_node_tf, axis=0),
                     tf.gather(w_tf, indices=global_mini_batch_indices, axis=0))

    def edge_h():
        log_term_train_edge = tf.log(1 - tf.multiply(ww, pi_tf))
        sum_log_term_train_edge = tf.reduce_sum(log_term_train_edge, axis=1, keepdims=True) + tf.log(1 - pi_0_tf)
        return tf.divide(tf.multiply(ww, tf.exp(sum_log_term_train_edge - log_term_train_edge)),
                         (1 - tf.exp(sum_log_term_train_edge)))

    def non_edge_h():
        return tf.divide(-ww, 1 - tf.multiply(ww, pi_tf))

    h = tf.cond(tf.equal(edge_tf, 1), edge_h, non_edge_h)  # mc_sample * mini_batch_size_tf * K

    #  Calculating G
    common_term = tf.concat((pi_tf*(1 - pi_tf), -pi_tf*(1 - pi_tf)), axis=0)  # 2 * K array
    gradient_pi_bar_tf = tf.multiply(common_term, tf.reduce_sum(h, axis=0))  # 2 * K array

    scale = tf.cond(tf.equal(edge_tf, 1), lambda: n_tf,
                    lambda: tf.divide(tf.multiply(n_tf, n_tf), tf.cast(mini_batch_size_tf, tf.float32)))
    scaled_gradient_pi_bar_tf = tf.multiply(scale, gradient_pi_bar_tf)

    pi_bar = tf.multiply(tf.concat((pi_tf, 1 - pi_tf), axis=0), sum_pi_bar_tf)  # 2 * K array
    xi = tf.random_normal(shape=(2, k_tf), mean=0., stddev=tf.sqrt(step_size_pi_tf), dtype=tf.float32)  # 2 * K array
    updated_pi_bar = tf.abs(pi_bar + tf.multiply(tf.sqrt(pi_bar), xi) +
                            tf.multiply(eta_0_by_1_tf - pi_bar + scaled_gradient_pi_bar_tf, tf.divide(step_size_pi_tf, 2)))

    updated_sum_pi_bar = tf.reduce_sum(updated_pi_bar, axis=0, keepdims=True)  # 1 * K array
    updated_pi = tf.divide(updated_pi_bar[0, :], updated_sum_pi_bar)  # 1* K array

    assign_pi_tf = pi_tf.assign(updated_pi)
    assign_sum_pi_bar_tf = sum_pi_bar_tf.assign(updated_sum_pi_bar)

    return [assign_pi_tf, assign_sum_pi_bar_tf]

