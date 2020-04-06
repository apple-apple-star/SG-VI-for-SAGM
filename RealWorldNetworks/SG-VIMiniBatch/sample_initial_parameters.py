import numpy as np
import tensorflow as tf
from util import Map

def sample_initial_parameters_pi_alpha(k, alpha, eta_0, eta_1):
    pi_bar_gamma_shape = np.float32(np.repeat([np.array([eta_0, eta_1])], [k], axis=0).T)  # 2*K matrix
    pi_bar = np.float32(np.random.gamma(pi_bar_gamma_shape, 1))

    sum_pi_bar = np.sum(pi_bar, axis=0).reshape((1, k))
    pi = np.divide(pi_bar[0, :], sum_pi_bar)

    pi_tf = tf.Variable(pi, dtype=tf.float32, name="pi_tf")
    sum_pi_bar_tf = tf.Variable(sum_pi_bar, dtype=tf.float32, name="sum_pi_bar_tf")

    alpha_tf = tf.Variable(alpha, dtype=tf.float32, name="alpha_tf")


    return pi_tf, sum_pi_bar_tf, alpha_tf

def sample_initial_parameters_w(k, n):
    sum_w_bar = np.ones((n, k))
    w = np.divide(np.ones((n, k)), k)

    return w, sum_w_bar

def sample_initial_parameters_model(k, n):
    sCur_phi_bar = np.zeros((n, k, 2))
    sCur_lambda_bar = np.zeros((2, k))
    sCur_tau = np.zeros((2, k))

    pi0 = 0.00005
    P0 = np.log(1 - pi0)
    tau = np.ones((2, k))

    param_lambda = np.ones((2, k))
    lambda_bar = np.log(np.exp(param_lambda) - 1)

    phi = np.random.randint(low=1, high=20, size=(n, k, 2))
    phi_bar = np.log(np.exp(phi) - 1)

    model = Map(sCur_phi_bar=sCur_phi_bar, sCur_lambda_bar=sCur_lambda_bar, sCur_tau=sCur_tau, pi0=pi0, P0=P0,
                tau=tau, lambda_bar=lambda_bar, phi_bar=phi_bar)

    return model