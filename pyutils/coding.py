import numpy as np

import sklearn.neighbors as neighbors


def llcEncoding(data, codebook, knn, kdtree_codebook=None, beta=1.0):

    n = data.shape[0]

    if kdtree_codebook is None:
        kdtree_codebook = neighbors.KDTree(codebook, leaf_size=4)

    _, idx = kdtree_codebook.query(data, knn)

    codes = []

    for i in range(n):

        z = codebook[idx[i], :] - np.repeat(data[i][np.newaxis, :], knn, axis=0)
        C = z.dot(z.T)
        C = C + beta * np.trace(C) * np.eye(knn)
        w = np.linalg.solve(C, np.ones((knn, 1)))
        w[w < 0] = 0.0
        w = w / np.sum(w)
        codes.append(zip(idx[i], w.ravel()))

    return codes
