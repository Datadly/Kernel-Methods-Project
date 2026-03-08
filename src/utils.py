import numpy as np

def standardize(X_train, X_test): # Does not leak test data info
    mean = np.mean(X_train, axis = 0)

    std = np.std(X_train, axis = 0)
    std[std == 0] = 1.0

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std 

    return X_train_std, X_test_std

def augment_data(X, y): # Augments dataset by adding mirror clones
    
    N = X.shape[0]
    
    X_imgs = X.reshape(N, 3, 32, 32)
    
    X_flipped = X_imgs[:, :, :, ::-1] # Horizontal flip
    
    X_flipped = X_flipped.reshape(N, 3072)
    
    X_aug = np.vstack([X, X_flipped])
    y_aug = np.concatenate([y, y])
    
    return X_aug, y_aug

def gaussian_blur(X_raw): 
    N = X_raw.shape[0]
    img = X_raw.reshape(N, 3, 32, 32)
    kernel = np.array([[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]]) / 16.0
    
    blurred = np.copy(img)

    for c in range(3): # Manual Convolution
        for i in range(1, 31):
            for j in range(1, 31):
                region = img[:, c, i-1:i+2, j-1:j+2]
                blurred[:, c, i, j] = np.sum(region * kernel, axis=(1, 2))
    return blurred.reshape(N, 3072)

def augment_data_v2(X, y):
    
    X_mirror = X.reshape(-1, 3, 32, 32)[:, :, :, ::-1].reshape(-1, 3072) # Apply mirror
    X_blur = gaussian_blur(X) # Apply Gaussian blur
    
    X_aug = np.vstack([X, X_mirror, X_blur])
    y_aug = np.concatenate([y, y, y])
    return X_aug, y_aug