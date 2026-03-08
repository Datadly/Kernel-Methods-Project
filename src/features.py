import numpy as np

def hog_descriptor(images, cell_size=8, n_bins=9): # HOG feature exaction
    
    N = images.shape[0]
    # Greying Images (ITU-R BT.601 standard : 0.299R + 0.587G + 0.114B)
    imgs_rgb = images.reshape(-1, 3, 32, 32)
    imgs_gray = 0.299 * imgs_rgb[:, 0] + 0.587 * imgs_rgb[:, 1] + 0.114 * imgs_rgb[:, 2]
    
    # Computing Gradients
    gx = np.zeros_like(imgs_gray)
    gy = np.zeros_like(imgs_gray)
    gx[:, :, 1:-1] = imgs_gray[:, :, 2:] - imgs_gray[:, :, :-2]
    gy[:, 1:-1, :] = imgs_gray[:, 2:, :] - imgs_gray[:, :-2, :]
    
    # Computing Magnitude and Angles
    magnitudes = np.sqrt(gx**2 + gy**2)
    orientations = np.arctan2(gy, gx) * (180 / np.pi) % 180 # 0 à 180°
    
    # Creating Histograms
    n_cells = 32 // cell_size
    hog_features = []
    
    for n in range(N):
        img_hists = []
        for i in range(n_cells):
            for j in range(n_cells):
                c_mag = magnitudes[n, i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                c_ori = orientations[n, i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                
                hist, _ = np.histogram(c_ori, bins=n_bins, range=(0, 180), weights=c_mag)
                img_hists.append(hist)
        
        # Normalization 
        feat = np.concatenate(img_hists)
        norm = np.linalg.norm(feat) + 1e-6
        hog_features.append(feat / norm)
        
    return np.array(hog_features)

def hog_descriptor_v2(X, cell_size=4, orientations=9): # HOG + L2 Lys Block Normalization + Max-Channel gradients

    N = X.shape[0]
    X_imgs = X.reshape(N, 3, 32, 32)
    
    n_cells = 32 // cell_size
    all_hogs = []

    for i in range(N):
        img = X_imgs[i] # (3, 32, 32)
        
        gx = np.zeros((3, 32, 32))
        gy = np.zeros((3, 32, 32))
        
        gx[:, :, 1:-1] = img[:, :, 2:] - img[:, :, :-2]
        gy[:, 1:-1, :] = img[:, 2:, :] - img[:, :-2, :]
        
        mag = np.sqrt(gx**2 + gy**2) # (3, 32, 32)
        
        best_ch = np.argmax(mag, axis=0) # Keep max channel (32, 32)
        
        grid_y, grid_x = np.indices((32, 32))
        final_mag = mag[best_ch, grid_y, grid_x]
        
        final_gx = gx[best_ch, grid_y, grid_x]
        final_gy = gy[best_ch, grid_y, grid_x]
        final_ang = np.arctan2(final_gy, final_gx) * (180 / np.pi) % 180

        bin_width = 180 / orientations
        cells_hist = np.zeros((n_cells, n_cells, orientations))
        
        for r in range(n_cells):
            for c in range(n_cells):
                m_c = final_mag[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                a_c = final_ang[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                
                bins = np.floor(a_c / bin_width).astype(int)
                bins = np.clip(bins, 0, orientations - 1)
                
                for b in range(orientations):
                    cells_hist[r, c, b] = np.sum(m_c[bins == b])

        # L2 Lys Normalization
        eps = 1e-5
        blocks = []
        for r in range(n_cells - 1):
            for c in range(n_cells - 1):
                block = cells_hist[r:r+2, c:c+2, :].flatten()
                
                norm = np.sqrt(np.sum(block**2) + eps**2)
                block /= norm
                
                block = np.minimum(block, 0.2)
                block /= (np.sqrt(np.sum(block**2) + eps**2))
                
                blocks.append(block)
        
        all_hogs.append(np.concatenate(blocks))

    return np.array(all_hogs)


def color_histogram(X, bins=16):
    N = X.shape[0]
    features = np.zeros((N, 3 * bins))

    for i in range(N):
        img = X[i]

        r = img[:1024]
        g = img[1024:2048]
        b = img[2048:]

        h_r, _ = np.histogram(r, bins=bins, range=(0, 255), density=True)
        h_g, _ = np.histogram(g, bins=bins, range=(0, 255), density=True)
        h_b, _ = np.histogram(b, bins=bins, range=(0, 255), density=True)

        features[i] = np.concatenate([h_r, h_g, h_b])
    
    return features