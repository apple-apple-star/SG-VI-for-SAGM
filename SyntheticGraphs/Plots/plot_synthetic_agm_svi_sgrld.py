# Without RMSProp (Riemann)
from numpy import load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np



K = [10, 20, 30, 40, 50]
n = "LineSearch_agm-75-4"
svi_mid = "seed0lr_lambda0.0lr_phi0.0lr_tau0.0riemann0elbo1mc5L20M20K"
svi_mid_reimann = "seed0lr_lambda0.0lr_phi0.0lr_tau0.0riemann1elbo1mc5L20M20K"
end_svi = "max iterations10000test_ratio0.1.npz"
sgrld_mid = "Seed0L20M20K"
end_sgrld = "burn_in5000samples5000test_ratio0.1lrw0.0lrpi0.0.npz"
###

svi_auc = []
svi_reimann_auc = []
sgrld_auc = []
svi_pp = []
svi_reimann_pp = []
sgrld_pp = []
i = 0
for k in K:
    curr_file_sgrld = '../Results/' + n + sgrld_mid + str(k) + end_sgrld
    curr_file_svi_reimann = '../Results/' + n + svi_mid_reimann + str(k) + end_svi
    curr_file_svi = '../Results/' + n + svi_mid + str(k) + end_svi

    data_svi_reimann = load(curr_file_svi_reimann)
    test_auc_svi_reimann = data_svi_reimann['TestAUCvector']
    svi_reimann_auc.append(test_auc_svi_reimann[-1])
    test_pp_svi_reimann = data_svi_reimann['TestPPvector']
    svi_reimann_pp.append(test_pp_svi_reimann[-1])


    data_sgrld = load(curr_file_sgrld)
    # test_auc_sgrld = data_sgrld['TestAUCvector']
    # sgrld_auc.append(test_auc_sgrld[-1])
    test_auc_sgrld = data_sgrld['AvgAUC']
    sgrld_auc.append(test_auc_sgrld)
    test_pp_sgrld = data_sgrld['TestPPvector']
    sgrld_pp.append(test_pp_sgrld[-1])


    data_svi = load(curr_file_svi)
    test_auc_svi = data_svi['TestAUCvector']
    svi_auc.append(test_auc_svi[-1])
    test_pp_svi = data_svi['TestPPvector']
    svi_pp.append(test_pp_svi[-1])

    i = i + 1


plt.figure(figsize=(12, 6))
# AUC
plt.subplot(1, 2, 1)
# plt.plot(svi_auc, linewidth=1.0, label='SGD-VI without Preconditioner')
plt.plot(svi_reimann_auc, linewidth=2.0, label='SG-VI with Preconditioner')
plt.plot(sgrld_auc, linewidth=2.0, label='SGRLD')
plt.tick_params(labelsize=14)
plt.ylabel('AUC-ROC', fontsize=14)
plt.xlabel('K', fontsize=14)
plt.legend(fontsize=14)
plt.ylim(0.8, 1.0)
plt.xticks(np.arange(5), ('10', '20', '30', '40', '50'))

# Perplexity
plt.subplot(1, 2, 2)
# plt.plot(svi_pp, linewidth=1.0, label='SGD-VI without Preconditioner')
plt.plot(svi_reimann_pp, linewidth=2.0, label='SG-VI with Preconditioner')
plt.plot(sgrld_pp, linewidth=2.0, label='SGRLD')
plt.tick_params(labelsize=14)
plt.ylabel('Perplexity', fontsize=14)
plt.xlabel('K', fontsize=14)
plt.legend(fontsize=14)
plt.ylim(1.0, 3.0)
plt.xticks(np.arange(5), ('10', '20', '30', '40', '50'))
#
# plt.show()

plt.savefig("synthetic_agm_graph_svi_sgrld.pdf")
plt.savefig("synthetic_agm_graph_svi_sgrld.png")