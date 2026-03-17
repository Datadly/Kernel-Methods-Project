import sys
import os
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.utils import standardize, augment_data, augment_data_v2
from src.features import hog_descriptor, hog_descriptor_v2, spatial_color_histogram, color_histogram, spatial_color_histogram_custom
from src.models import KernelSVM, KernelRidgeRegression, KRR_Custom
from src.kernels import rbf_kernel, laplacian_kernel

np.random.seed(42)


def main(): 
    
    # --- Data Loading ---
    data_path = 'data/'
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(data_path, 'Xtr.csv'), header=None)
    df_test = pd.read_csv(os.path.join(data_path, 'Xte.csv'), header=None)
    y_train_raw = pd.read_csv(os.path.join(data_path, 'Ytr.csv'))

    X_all_raw = df_train.iloc[:, :3072].values
    X_test_raw = df_test.iloc[:, :3072].values
    y_all_raw = y_train_raw.iloc[:, 1].values
    print(f"Train: {X_all_raw.shape}, Test: {X_test_raw.shape}")

    # --- Data Augmentation ---
    X_aug, y_aug = augment_data_v2(X_all_raw, y_all_raw)

    # ---  Feature Extraction ---
    conf_grid = (3, 3) 
    conf_bins = 16
    conf_orient = 9

    print(f"Extracting HOG (Train Aug: {X_aug.shape[0]}, Test: {X_test_raw.shape[0]})...")
    X_train_hog = hog_descriptor_v2(X_aug, cell_size=4, orientations=conf_orient)
    X_test_hog = hog_descriptor_v2(X_test_raw, cell_size=4, orientations=conf_orient)
    

    # --- Color Extraction ---
    print("Extracting Global Colors (bins=16)...")
    X_train_sp = spatial_color_histogram_custom(X_aug, bins=conf_bins, grid=conf_grid)
    X_test_sp = spatial_color_histogram_custom(X_test_raw, bins=conf_bins, grid=conf_grid)

    # --- Standadization ---
    print("Fusing HOG and Color features...")
    X_tr_hog_std, X_te_hog_std = standardize(X_train_hog, X_test_hog)
    X_tr_sp_std, X_te_sp_std = standardize(X_train_sp, X_test_sp)

    # --- Parameters ---
    sigma_hog = 30    
    sigma_color = 60 
    lambda_reg = 1e-8
    n_classes = 10

    # --- Compute Gram Matrices ---
    K_tr_hog = rbf_kernel(X_tr_hog_std, X_tr_hog_std, sigma=sigma_hog)
    K_te_hog = rbf_kernel(X_te_hog_std, X_tr_hog_std, sigma=sigma_hog)

    K_tr_sp = rbf_kernel(X_tr_sp_std, X_tr_sp_std, sigma=sigma_color)
    K_te_sp = rbf_kernel(X_te_sp_std, X_tr_sp_std, sigma=sigma_color)

    # --- Fusion ---
    K_train_total = K_tr_hog + K_tr_sp
    K_test_total = K_te_hog + K_te_sp
    
    # --- One Hot Encoding ---
    Y_multi = np.full((len(y_aug), n_classes), -1.0) 
    for idx, val in enumerate(y_aug):
        Y_multi[idx, val] = 1.0

    # --- Model (Laplacian KRR) ---
    print(f"Training RBF KRR (sigma={sigma_hog}, lambda={lambda_reg})...")
    n = K_train_total.shape[0]
    A = K_train_total + lambda_reg * n * np.eye(n)
    alpha = np.linalg.solve(A, Y_multi)

    # --- Final Prediction (On whole dataset) ---
    print("Predicting ...")
    test_scores = K_test_total @ alpha
    y_pred_final = np.argmax(test_scores, axis=1)

    # ---  Kaggle submission file ---
    submission = pd.DataFrame({
        'Id': np.arange(1, len(y_pred_final) + 1),
        'Prediction': y_pred_final
    })
    
    output_file = 'Yte_pred.csv'
    submission.to_csv(output_file, index=False)
    print(f"Submission File saved as : {output_file}")
    

if __name__ == "__main__":
    main()