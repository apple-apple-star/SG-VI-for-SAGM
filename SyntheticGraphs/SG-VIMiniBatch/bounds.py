
import tensorflow as tf


def probability_bound(var, matrix_ones):
    lesser_lim = tf.constant(0.00001, dtype=tf.float64)
    greater_lim = tf.constant(1 - 0.00001, dtype=tf.float64)
    comparison_var_less = tf.less(var, lesser_lim)
    comparison_var_greater = tf.greater(var, greater_lim)
    var = tf.where(comparison_var_less, matrix_ones * lesser_lim, var)
    var = tf.where(comparison_var_greater, matrix_ones * greater_lim, var)

    return var

def exponentional_bound(var, matrix_ones):
    lower_value_positive = tf.constant(1e-1, dtype=tf.float64)
    truncate_lesser_lim = tf.log(tf.exp(lower_value_positive)-1)
    truncate_greater_lim = tf.constant(1e+300, dtype=tf.float64)
    comparison_var_less = tf.less(var, truncate_lesser_lim)
    comparison_var_greater = tf.greater(var, truncate_greater_lim)
    var = tf.where(comparison_var_less, matrix_ones * truncate_lesser_lim, var)
    var = tf.where(comparison_var_greater, matrix_ones * truncate_greater_lim, var)
    return var

def greater_log_exp_bound(matrix, bar_matrix):
    # matrix = tf.log(tf.exp(bar_matrix)+1)
    log_exp_greater_lim = 1e+2
    comparison_var_greater = tf.greater(bar_matrix, log_exp_greater_lim)
    var = tf.where(comparison_var_greater, bar_matrix, matrix)
    return var

def replace_zero(to_replace_matrix, matrix):
    comparison_var_equal = tf.equal(matrix, 0.0)
    var = tf.where(comparison_var_equal, matrix, to_replace_matrix)
    return var
