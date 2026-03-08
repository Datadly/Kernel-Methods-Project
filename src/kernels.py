import numpy as np

def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

def rbf_kernel(X1, X2, sigma=1):
    norm1 = np.sum(X1**2, axis=1).reshape(-1,1)
    norm2 = np.sum(X2**2, axis=1).reshape(-1,1)
    dist_sq = norm1 + norm2.T - 2 * np.dot(X1, X2.T)
    return np.exp(-dist_sq / (2 * sigma**2))

def laplacian_kernel(X1, X2, sigma=10.0):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        dist_l1 = np.sum(np.abs(X1[i] - X2), axis=1)
        K[i, :] = np.exp(-dist_l1 / sigma)
    return K

