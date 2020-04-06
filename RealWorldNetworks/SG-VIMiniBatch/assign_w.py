import tensorflow as tf

from bounds import greater_log_exp_bound

def assign_w(phi_bar_tf):
    phi = tf.log(tf.exp(phi_bar_tf) + 1)
    phi = greater_log_exp_bound(phi, phi_bar_tf)
    w = tf.divide(phi[:, :, 0], phi[:, :, 0] + phi[:, :, 1])
    return w