from numpy import load
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab


mini_batch_size = [10, 50, 100, 150, 200]
max_iter = [60000, 50000, 40000, 30000, 20000]
burn_in = [50000, 40000, 30000, 20000, 10000]
# ca-HepPhlr_lambda1.0lr_phi1.0lr_tau1.0riemann1elbo0mc1L50M50K50max iterations50000test_ratio0.1.npz
network = ['ca-HepPh', 'FA', 'Reuters', 'enron']
svi_mid = "lr_lambda1.0lr_phi1.0lr_tau1.0riemann1elbo0mc1"
end_svi = "test_ratio0.1.npz"
end_sgrld = "samples10000test_ratio0.1lrw0.001lrpi0.001.npz"
###

# plt.suptitle('AUC vs Time (sec)', fontsize=10)
# plt.subplots_adjust(hspace=0.5)

plt.figure(figsize=(20, 14))

i = 1
for n in network:
    for m, l, o in zip(mini_batch_size, max_iter, burn_in):
        plt.subplot(4, 5, i)
        curr_file_svi = '../Results/'+n+svi_mid+'L'+str(m)+'M'+str(m)+'K50max iterations'+str(l)+end_svi
        curr_file_sgrld = '../Results/' + n + "Seed0L" + str(m) + "M" + str(m) + "K50burn_in" + str(o) + end_sgrld
        data_svi = load(curr_file_svi)
        data_sgrld = load(curr_file_sgrld)
        test_auc_svi = data_svi['TestAUCvector']
        test_auc_sgrld = data_sgrld['TestAUCvector']
        plt.plot(test_auc_svi, linewidth=2.0, label='SG-VI')
        plt.plot(test_auc_sgrld, linewidth=2.0, label='SGRLD')
        plt.tick_params(labelsize=14)
        if i == 1:
            plt.title('Mini-batch : 10', fontsize=14)
        elif i == 2:
            plt.title('Mini-batch : 50', fontsize=14)
        elif i == 3:
            plt.title('Mini-batch : 100', fontsize=14)
        elif i == 4:
            plt.title('Mini-batch : 150', fontsize=14)
        elif i == 5:
            plt.title('Mini-batch : 200', fontsize=14)
        if i % 5 == 1:
            plt.ylabel(str(n), fontsize=14)
            plt.yticks([0.2, 0.4, 0.6, 0.8])
        else:
            plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        plt.xticks(np.arange(0, len(test_auc_sgrld)+1, step=100))
        pylab.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
        plt.legend(fontsize=14)
        # plt.yticks([0.2, 0.4, 0.6, 0.8, 0.9])
        i = i+1


# plt.show()
plt.savefig("larger_graph_svi_sgrld.pdf")