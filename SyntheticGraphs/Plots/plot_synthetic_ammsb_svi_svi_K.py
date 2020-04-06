from numpy import load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


K = [10, 20, 30, 40]
n = "LineSearch_assort-75-4"
svi_mid = "seed0lr_lambda0.0lr_phi0.0lr_tau0.0riemann0elbo1mc5L20M20K"
svi_mid_reimann = "seed0lr_lambda0.0lr_phi0.0lr_tau0.0riemann1elbo1mc5L20M20K"
end_svi = "max iterations10000test_ratio0.1_all_auc.npz"
sgrld_mid = "Seed0L20M20K"
end_sgrld = "burn_in5000samples5000test_ratio0.1lrw0.0lrpi0.0_all_auc.npz"
###


plt.figure(figsize=(16, 10))

# With Preconditioner

i = 1
for k in K:
    plt.subplot(2, 2, i)
    curr_file_svi_reimann = '../Results/' + n + svi_mid_reimann + str(k) + end_svi
    data_svi_reimann = load(curr_file_svi_reimann)
    all_auc_svi_reimann = data_svi_reimann['all_auc']
    curr_file_svi = '../Results/' + n + svi_mid + str(k) + end_svi
    data_svi = load(curr_file_svi)
    all_auc_svi = data_svi['all_auc']
    plt.hist([all_auc_svi_reimann, all_auc_svi], bins=10,
             label=['With preconditioner', 'Without preconditioner'])
    plt.tick_params(labelsize=18)
    plt.ylabel('K = '+str(k), fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.xlabel('AUC-ROC', fontsize=18)
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    i = i + 1

# plt.show()

plt.savefig("synthetic_ammsb_graph_svi_svi_K.pdf")
plt.savefig("synthetic_ammsb_graph_svi_svi_K.png")