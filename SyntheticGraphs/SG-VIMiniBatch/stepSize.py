import tensorflow as tf

def stepSize(iteration, sPrev, gradient, eta_tf):
    sCur = 0.1 * tf.pow(gradient, 2) + 0.9 * sPrev
    step = tf.divide(eta_tf * tf.pow(iteration, (-0.5 + 1e-16)), (1 + tf.sqrt(sCur)))

    return [step, sCur]