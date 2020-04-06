import numpy as np
from util import Map

def sample_initial_parameters_pi_alpha(k, eta_0, eta_1):
    pi_bar_gamma_shape = np.float32(np.repeat([np.array([eta_0, eta_1])], [k], axis=0).T)  # 2*K matrix
    pi_bar = np.float32(np.random.gamma(pi_bar_gamma_shape, 1))

    sum_pi_bar = np.sum(pi_bar, axis=0).reshape((1, k))
    pi = np.divide(pi_bar[0, :], sum_pi_bar)

    alpha = np.float32(np.ones((1, k)))

    return pi, sum_pi_bar, alpha

def sample_initial_parameters_w(k, n):
    sum_w_bar = np.ones((n, k))
    w = np.divide(np.ones((n, k)), k)

    return w, sum_w_bar

def sample_initial_parameters_model(k, n, eta_0, eta_1):
    w, sum_w_bar = sample_initial_parameters_w(k, n)

    pi, sum_pi_bar, alpha = sample_initial_parameters_pi_alpha(k, eta_0, eta_1)

    model = Map(w=w, sum_w_bar=sum_w_bar, pi=pi, sum_pi_bar=sum_pi_bar, alpha=alpha)

    return model
