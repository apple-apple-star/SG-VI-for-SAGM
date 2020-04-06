from numpy import load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


K = [10, 20, 30, 40, 50]
n = "LineSearch_agm-75-4"
svi_mid = "seed0lr_lambda0.0lr_phi0.0lr_tau0.0riemann0elbo1mc5L20M20K"
svi_mid_reimann = "seed0lr_lambda0.0lr_phi0.0lr_tau0.0riemann1elbo1mc5L20M20K"
end_svi = "max iterations10000test_ratio0.1_all_auc.npz"
sgrld_mid = "Seed0L20M20K"
end_sgrld = "burn_in5000samples5000test_ratio0.1lrw0.0lrpi0.0_all_auc.npz"
###


plt.figure(figsize=(10, 4))

# With Preconditioner
plt.subplot(1, 2, 1)
i = 0
all_auc = np.zeros((5, 64))
for k in K:
    curr_file_svi_reimann = '../Results/' + n + svi_mid_reimann + str(k) + end_svi
    data_svi_reimann = load(curr_file_svi_reimann)
    all_auc[i, :] = data_svi_reimann['all_auc']
    i = i + 1

plt.hist([all_auc[0, :], all_auc[1, :], all_auc[2, :], all_auc[3, :], all_auc[4, :]],
         bins=10, label=['K = 10', 'K = 20', 'K = 30', 'K = 40', 'K = 50'])
plt.legend(loc='upper left', fontsize=8)
plt.xlabel('AUC-ROC', fontsize=8)
plt.title('SGD-VI with preconditioner')

# Without Preconditioner
plt.subplot(1, 2, 2)
i = 0
all_auc = np.zeros((5, 64))
for k in K:
    curr_file_svi = '../Results/' + n + svi_mid + str(k) + end_svi
    data_svi = load(curr_file_svi)
    all_auc[i, :] = data_svi['all_auc']
    i = i + 1

plt.hist([all_auc[0, :], all_auc[1, :], all_auc[2, :], all_auc[3, :], all_auc[4, :]],
         bins=10, label=['K = 10', 'K = 20', 'K = 30', 'K = 40', 'K = 50'])
plt.legend(loc='upper left', fontsize=8)
plt.xlabel('AUC-ROC', fontsize=8)
plt.title('SGD-VI without preconditioner')


# plt.show()

plt.savefig("synthetic_agm_graph_svi_svi.pdf")
plt.savefig("synthetic_agm_graph_svi_svi.png")