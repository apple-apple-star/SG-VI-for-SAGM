# Without RMSProp (Riemann)
from numpy import load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


mini_batch_size = [10, 50, 100, 150, 200]
max_iter = [60000, 50000, 40000, 30000, 20000]
# ca-HepPhlr_lambda1.0lr_phi1.0lr_tau1.0riemann1elbo0mc1L50M50K50max iterations50000test_ratio0.1.npz
network = ['ca-HepPh', 'FA', 'Reuters']
mid = "lr_lambda1.0lr_phi1.0lr_tau1.0riemann1elbo0mc1"
end = "test_ratio0.1.npz"
###

# plt.suptitle('AUC vs Time (sec)', fontsize=10)
plt.subplots_adjust(hspace=0.5)

i = 1
for n in network:
    plt.subplot(3, 2, i)
    for m, l in zip(mini_batch_size, max_iter):
        curr_file = '../Results/'+n+mid+'L'+str(m)+'M'+str(m)+'K50max iterations'+str(l)+end
        data = load(curr_file)
        test_auc = data['TestAUCvector']
        time_vec = data['TimeVector']
        plt.plot(time_vec, test_auc, linewidth=1.0)
        plt.tick_params(labelsize=4)
        if i == 1:
            plt.legend(["Mini-batch : 10", "Mini-batch : 50", "Mini-batch : 100",
                        "Mini-batch : 150", "Mini-batch : 200"],
                       bbox_to_anchor=(1.1, 1.5), ncol=5, loc='upper center', prop={'size': 6})
        plt.ylabel(str(n), fontsize=8)
        if i == 5:
            plt.xlabel('AUC-ROC', fontsize=8)

    i = i + 1
    plt.subplot(3, 2, i)
    for m, l in zip(mini_batch_size, max_iter):
        curr_file = '../Results/'+n+mid+'L'+str(m)+'M'+str(m)+'K50max iterations'+str(l)+end
        data = load(curr_file)
        test_auc = data['TestPPvector']
        time_vec = data['TimeVector']
        plt.plot(time_vec, test_auc, linewidth=1.0)
        plt.tick_params(labelsize=4)
        if i == 6:
            plt.xlabel('Perplexity', fontsize=8)

    i = i + 1

plt.show()
# plt.savefig("larger_graph_svi.pdf")