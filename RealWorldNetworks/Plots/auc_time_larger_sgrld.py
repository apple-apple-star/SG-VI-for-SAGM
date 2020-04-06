# Without RMSProp (Riemann)
from numpy import load



mini_batch_size = [10, 50, 100, 150, 200]
burn_in = [50000, 40000, 30000, 20000, 10000]
# ca-HepPhSeed0L10M10K50burn_in50000samples10000test_ratio0.1lrw0.001lrpi0.001.npz
network = ['ca-HepPh', 'FA', 'Reuters', 'enron']
end = "samples10000test_ratio0.1lrw0.001lrpi0.001.npz"
###

for n in network:
    print(n)
    for m, l in zip(mini_batch_size, burn_in):
        curr_file = '../Results/' + n + "Seed0L" + str(m) + "M" + str(m) + "K50burn_in" + str(l) + end
        print(m)
        data = load(curr_file)
        test_auc = data['AvgAUC']
        time_vec = data['TimeVector']
        print("Time taken = ", time_vec[-1])
        print("AUC-ROC = ", test_auc[0])
