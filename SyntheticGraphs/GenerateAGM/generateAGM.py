import numpy as np
from scipy import sparse, io

k = 4
n = 75
backgroundProb = 0.00005
p0 = np.log(1 - backgroundProb)
p = np.ones((1, k)) * 0.8
### Community assignment matrix
Z = np.zeros((n, k))
Z[1:25, 0] = 1
Z[20:45, 1] = 1
Z[40:55, 2] = 1
Z[50:75, 3] = 1
#####
log_p = np.log(1-p)
P = np.identity(k) * log_p
ZPZ = np.matmul(np.matmul(Z, P), np.transpose(Z))
EdgeProb = 1 - np.exp(ZPZ + p0)

adjMat = np.zeros((n, n))
adjMat[np.random.uniform(0, 1, (n, n)) < EdgeProb] = 1

adjMat_UTM = np.triu(adjMat, k=1)

sparse_adjMat_UTM = sparse.csr_matrix(adjMat_UTM == 1)

outfile_network = '../Datasets/agm-75-4.npz'
sparse.save_npz(outfile_network, sparse_adjMat_UTM)


outfile_groundtruth = '../Datasets/agm-75-4-Z.npz'
np.savez(outfile_groundtruth, groundtruth=Z)
