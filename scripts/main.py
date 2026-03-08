import sys
import os
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.utils import standardize, augment_data, augment_data_v2
from src.features import hog_descriptor, hog_descriptor_v2
from src.models import KernelSVM, KernelRidgeRegression, KRR_Custom
from src.kernels import rbf_kernel, laplacian_kernel

np.random.seed(42)


def main(): 
    
    # --- Data Loading ---
    data_path = 'data/'
    print("🚀 Loading data...")
    df_train = pd.read_csv(os.path.join(data_path, 'Xtr.csv'), header=None)
    df_test = pd.read_csv(os.path.join(data_path, 'Xte.csv'), header=None)
    y_train_raw = pd.read_csv(os.path.join(data_path, 'Ytr.csv'))

    X_all_raw = df_train.iloc[:, :3072].values
    X_test_raw = df_test.iloc[:, :3072].values
    y_all_raw = y_train_raw.iloc[:, 1].values
    print(f"Train: {X_all_raw.shape}, Test: {X_test_raw.shape}")

    # --- Data Augmentation ---
    X_aug, y_aug = augment_data_v2(X_all_raw, y_all_raw)

    # --- HOG Feature Extraction ---
    print(f"feat: Extracting HOG (Train Aug: {X_aug.shape[0]}, Test: {X_test_raw.shape[0]})...")
    X_train_hog = hog_descriptor_v2(X_aug, cell_size=4, orientations=9)
    X_test_hog = hog_descriptor_v2(X_test_raw, cell_size=4, orientations=9)

    # --- Standardization ---
    X_train_std, X_test_std = standardize(X_train_hog, X_test_hog)

    # --- Parameters ---
    sigma_best = 700
    lam_best = 1e-12
    n_classes = 10

    print(f"🏋️ Training Laplacian KRR (sigma={sigma_best}, lambda={lam_best})...")
    
    # --- One Hot Encoding ---
    Y_multi = np.full((len(y_aug), n_classes), -1.0) 
    for idx, val in enumerate(y_aug):
        Y_multi[idx, val] = 1.0

    # --- Model (Laplacian KRR) ---
    final_model = KRR_Custom(sigma=sigma_best, lambd=lam_best, kernel_func=laplacian_kernel)
    final_model.fit(X_train_std, Y_multi)

    # --- Final Prediction (On whole dataset) ---
    print("Predicting ...")
    test_scores = final_model.predict(X_test_std)
    y_pred_final = np.argmax(test_scores, axis=1)

    # ---  Kaggle submission file ---
    submission = pd.DataFrame({
        'Id': np.arange(1, len(y_pred_final) + 1),
        'Prediction': y_pred_final
    })
    
    output_file = 'Yte_pred.csv'
    submission.to_csv(output_file, index=False)
    print(f"✅ Submission File saved as : {output_file}")
    

if __name__ == "__main__":
    main()