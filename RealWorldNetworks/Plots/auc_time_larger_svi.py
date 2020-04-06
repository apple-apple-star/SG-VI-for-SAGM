# Without RMSProp (Riemann)
from numpy import load



mini_batch_size = [10, 50, 100, 150, 200]
max_iter = [60000, 50000, 40000, 30000, 20000]
# ca-HepPhSeed0L10M10K50burn_in50000samples10000test_ratio0.1lrw0.001lrpi0.001.npz
network = ['ca-HepPh', 'FA', 'Reuters', 'enron']
mid = "lr_lambda1.0lr_phi1.0lr_tau1.0riemann1elbo0mc1"
end = "test_ratio0.1.npz"
###

for n in network:
    print(n)
    for m, l in zip(mini_batch_size, max_iter):
        curr_file = '../Results/' + n + mid + 'L' + str(m) + 'M' + str(m) + 'K50max iterations' + str(l) + end
        print(m)
        data = load(curr_file)
        time_vec = data['TimeVector']
        auc_vec = data['TestAUCvector']
        print("Time taken = ", time_vec[-1])
        print("AUC = ", auc_vec[-1])
