from numpy import load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

mc = [1, 50]
# mc = [1, 5, 10, 50]
max_iter = [10000]
n = 'enron'
mid_svi = "lr_lambda1.0lr_phi1.0lr_tau1.0riemann1elbo0mc"
end_svi = "L50M50K50max iterations10000test_ratio0.1.npz"
###

for m in mc:
    curr_file_svi = '../Results/' + n + mid_svi + str(m) + end_svi
    data_svi = load(curr_file_svi)
    test_auc_svi = data_svi['TestAUCvector']
    time = data_svi['TimeVector']
    print(time[-1])
    plt.plot(test_auc_svi, linewidth=2.0, label='Monte Carlo samples : '+ str(m))
plt.tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('AUC-ROC', fontsize=14)



# plt.show()
plt.savefig("enron_mc_svi.pdf")