import tensorflow as tf

from bounds import probability_bound, greater_log_exp_bound

def get_pp(log_sum_probs, true_y):
    num_test_links = tf.shape(true_y)[0]
    non_edge = tf.gather(log_sum_probs, tf.where(tf.equal(true_y, False)))
    edge = tf.gather(log_sum_probs, tf.where(tf.equal(true_y, True)))
    sum_test_ll = tf.reduce_sum(non_edge, 0) + tf.reduce_sum(tf.log(1-tf.exp(edge)), 0)
    pp = tf.exp(tf.divide(-sum_test_ll, tf.cast(num_test_links, tf.float64)))
    return [pp, sum_test_ll]



def get_roc_auc(num_probs, roc_y):
    count_nonzero = tf.count_nonzero(roc_y)
    count_zero = tf.count_nonzero(tf.logical_not(roc_y))
    stack_x = tf.divide(tf.cumsum(tf.cast(tf.equal(roc_y, False), tf.float64)), tf.cast(count_zero, tf.float64))
    stack_y = tf.divide(tf.cumsum(tf.cast(tf.equal(roc_y, True), tf.float64)), tf.cast(count_nonzero, tf.float64))
    a = tf.slice(stack_x, begin=[1], size=[num_probs-1]) - tf.slice(stack_x, begin=[0], size=[num_probs-1])
    b = tf.slice(stack_y, begin=[1], size=[num_probs-1])
    c = tf.reduce_sum(tf.multiply(a, b))
    return c

def calculate_auc_pp(phi_bar_tf, lambda_bar_tf, tau_tf, eta_0_by_1_tf, beta_0_tf, beta_1_tf, test_pair_tf,
                     test_links_tf, pi_0_tf, elbo_tf, scale_elbo_tf):
    phi = tf.log(tf.exp(phi_bar_tf) + 1)
    phi = greater_log_exp_bound(phi, phi_bar_tf)
    lambd = tf.log(tf.exp(lambda_bar_tf) + 1)
    lambd = greater_log_exp_bound(lambd, lambda_bar_tf)  # 2 * K
    w_tf = tf.divide(phi[:, :, 0], phi[:, :, 0] + phi[:, :, 1])
    w_tf = probability_bound(w_tf, tf.ones(tf.shape(w_tf), tf.float64))
    pi_tf = tf.divide(lambd[0, :], lambd[0, :] + lambd[1, :])
    pi_tf = probability_bound(pi_tf, tf.ones(tf.shape(pi_tf), tf.float64))
    indices_1 = tf.gather(test_pair_tf, indices=0, axis=0)
    indices_2 = tf.gather(test_pair_tf, indices=1, axis=0)
    start_node_prob = tf.gather(w_tf, indices=indices_1, axis=0)
    end_node_prob = tf.gather(w_tf, indices=indices_2, axis=0)
    log_sum_test = tf.log(1 - pi_0_tf) + tf.reduce_sum(
        tf.log(1 - tf.multiply(tf.multiply(start_node_prob, end_node_prob), pi_tf)), axis=1)
    # PP
    [pp, sum_test_ll] = get_pp(log_sum_test, test_links_tf)
    # AUC
    curr_test_pair_prob = 1 - tf.exp(log_sum_test)
    val, idx = tf.nn.top_k(curr_test_pair_prob, k=tf.shape(curr_test_pair_prob)[0])
    roc_y = tf.gather(test_links_tf, indices=idx)
    num_probs = tf.shape(curr_test_pair_prob)[0]
    auc_roc_calculated = get_roc_auc(num_probs, roc_y)

    # ELBO
    elbo = tf.cond(tf.equal(elbo_tf, 1), lambda: compute_elbo(sum_test_ll, phi, lambd, tau_tf, eta_0_by_1_tf,
                                                              beta_0_tf, beta_1_tf, scale_elbo_tf),
                   lambda: tf.cast(0, dtype=tf.float64))

    return [auc_roc_calculated, pp, elbo]

def compute_elbo(sum_test_ll, phi, lambd, tau_tf, eta_0_by_1_tf, beta_0_tf, beta_1_tf, scale_elbo_tf):
    # phi
    psi_phi = tf.digamma(phi)
    sum_phi = tf.reduce_sum(phi, axis=2)
    psi_sum_phi = tf.digamma(sum_phi)

    # lambda
    psi_lambda = tf.digamma(lambd)
    sum_lambda = tf.reduce_sum(lambd, axis=0)
    psi_sum_lambda = tf.digamma(sum_lambda)

    # tau
    psi_tau_0 = tf.digamma(tau_tf[0, :])
    psi_log_tau = psi_tau_0 - tf.log(tau_tf[1, :])
    tau_by_tau = tf.divide(tau_tf[0, :], tau_tf[1, :])

    # # Likelihood
    llhd = scale_elbo_tf*sum_test_ll
    # phi_term
    phi_term = tf.reduce_sum(psi_log_tau + (tau_by_tau - 1)*(psi_phi[:, :, 0] - psi_sum_phi))
    # lambda_term
    lambda_term = tf.reduce_sum(tf.lgamma(tf.reduce_sum(eta_0_by_1_tf)) - tf.reduce_sum(tf.lgamma(eta_0_by_1_tf)) + \
                  tf.reduce_sum((eta_0_by_1_tf - 1) * psi_lambda, axis=0) - \
                  (tf.reduce_sum(eta_0_by_1_tf) - 2)*psi_sum_lambda)
    # tau_term
    tau_term = tf.reduce_sum(beta_0_tf*tf.log(beta_1_tf) - tf.lgamma(beta_0_tf) +
                             (beta_0_tf - 1)*psi_log_tau - beta_1_tf*tau_by_tau)

    # phi_entropy_term
    phi_entropy_term = tf.reduce_sum(tf.lgamma(sum_phi) - tf.reduce_sum(tf.lgamma(phi), axis=2) + \
                       tf.reduce_sum((phi - 1)*psi_phi, axis=2) - (sum_phi - 2)*psi_sum_phi)
    # lambda_entropy_term
    lambda_entropy_term = tf.reduce_sum(tf.lgamma(sum_lambda) - tf.reduce_sum(tf.lgamma(lambd), axis=0) + \
                          tf.reduce_sum((lambd - 1)*psi_lambda, axis=0) - (sum_lambda - 2)*psi_sum_lambda)
    # tau_entropy_term
    tau_entropy_term = tf.reduce_sum(tau_tf[0, :]*(tf.digamma(tau_tf[0, :]) - 1) - tf.lgamma(tau_tf[0, :]) -
                                     psi_log_tau)
    entropy_term = phi_entropy_term + lambda_entropy_term + tau_entropy_term

    elbo = llhd + phi_term + lambda_term + tau_term - entropy_term

    return elbo