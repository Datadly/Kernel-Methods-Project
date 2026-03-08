import numpy as np
from .kernels import rbf_kernel

class KernelRidgeRegression:
    def __init__(self, sigma=1, lambd=0.1):
        self.sigma = sigma
        self.lambd = lambd
        self.alpha = None
        self.X_train = None

    def fit(self, X, Y_matrix):
        self.X_train = X
        n = X.shape[0]
        K = rbf_kernel(X, X, sigma=self.sigma)
        
        A = K + self.lambd * n * np.eye(n)
        self.alpha = np.linalg.solve(A, Y_matrix)

    def predict(self, X):
        K_test = rbf_kernel(X, self.X_train, sigma=self.sigma)
        return K_test @ self.alpha
    


class KRR_Custom:
    def __init__(self, sigma, lambd, kernel_func):
        self.sigma = sigma
        self.lambd = lambd
        self.kernel_func = kernel_func
        
    def fit(self, X, Y):
        self.X_train = X
        n = X.shape[0]
        K = self.kernel_func(X, X, sigma=self.sigma)
        A = K + self.lambd * n * np.eye(n)
        self.alpha = np.linalg.solve(A, Y)
        
    def predict(self, X):
        K_test = self.kernel_func(X, self.X_train, sigma=self.sigma)
        return K_test @ self.alpha

    

class KernelSVM:
    def __init__(self, func, C=1, epochs=100):
        self.func = func
        self.C = C
        self.epochs = epochs

        self.alpha = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples)

        K = self.func(X, X)

        lam = 1 / (self.C * n_samples)

        for t in range(1, self.epochs + 1):
            indices = np.random.permutation(n_samples)

            for i in indices:
                pred = np.dot(self.alpha, K[:, i])
                lr = 1 / (lam * t * n_samples)

                if y[i] * pred <= 1:
                    self.alpha *= (1 - lr * lam)
                    self.alpha[i] += lr * y[i]
                else:
                    self.alpha[i] *= (1 - lr * lam)
    
    def predict(self, X):
        K_test = self.func(X, self.X_train)
        return K_test @ self.alpha

