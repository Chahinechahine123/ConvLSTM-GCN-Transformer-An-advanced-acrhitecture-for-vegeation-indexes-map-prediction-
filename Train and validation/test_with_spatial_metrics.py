from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy import sparse
from skimage.metrics import structural_similarity as ssim

# === Chemins ===
MODEL_PATH = "brute_data/ConvLstm_Gcn.keras"
DATA_DIR   = "brute_data/modis"
OUT_DIR    = "brute_data/modis/test_outputs_sentinel"
os.makedirs(OUT_DIR, exist_ok=True)

# === Chargement données ===
model  = tf.keras.models.load_model(MODEL_PATH)
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))    # (N, 9, H, W, 1)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))    # (N, H, W, 1)

print("Shapes – X:", X_test.shape, "| y:", y_test.shape)

# === Prédiction ===
y_pred = model.predict(X_test, batch_size=1, verbose=1)   # (N, H, W, 1)

# === Métriques globales ===
y_true_flat = y_test.reshape(len(y_test), -1)
y_pred_flat = y_pred.reshape(len(y_pred), -1)

rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
mae  = mean_absolute_error(y_true_flat, y_pred_flat)
nse  = 1 - np.sum((y_true_flat - y_pred_flat)**2) / np.sum((y_true_flat - y_true_flat.mean())**2)

# === Calcul du SSIM pour chaque échantillon ===
ssim_values = []
for i in range(len(y_test)):
    ssim_val = ssim(
        y_test[i, :, :, 0], 
        y_pred[i, :, :, 0],
        data_range=y_test[i, :, :, 0].max() - y_test[i, :, :, 0].min(),
        win_size=7  # taille de fenêtre pour SSIM
    )
    ssim_values.append(ssim_val)
ssim_mean = np.mean(ssim_values)

# === Calcul du Moran's I pour chaque échantillon ===
def compute_morans_i(residual_map):
    """Calcule le Moran's I pour une carte de résidus 2D."""
    H, W = residual_map.shape
    n = H * W
    
    # Créer une matrice de poids spatiaux (4-voisins)
    indices = np.arange(n).reshape(H, W)
    rows, cols = [], [], []
    
    for i in range(H):
        for j in range(W):
            idx = indices[i, j]
            # Voisins haut
            if i > 0:
                rows.append(idx)
                cols.append(indices[i-1, j])
            # Voisins bas
            if i < H-1:
                rows.append(idx)
                cols.append(indices[i+1, j])
            # Voisins gauche
            if j > 0:
                rows.append(idx)
                cols.append(indices[i, j-1])
            # Voisins droite
            if j < W-1:
                rows.append(idx)
                cols.append(indices[i, j+1])
    
    # Matrice de contiguïté binaire
    W_mat = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    
    # Centrer les résidus
    residuals_flat = residual_map.flatten()
    mean_r = np.mean(residuals_flat)
    residuals_centered = residuals_flat - mean_r
    
    # Calcul du Moran's I
    numerator = np.sum(W_mat.multiply(np.outer(residuals_centered, residuals_centered)))
    denominator = np.sum(residuals_centered**2)
    W_sum = W_mat.sum()
    
    if denominator == 0 or W_sum == 0:
        return 0.0
    
    morans_i = (n / W_sum) * (numerator / denominator)
    return morans_i

morans_i_values = []
for i in range(len(y_test)):
    residuals = y_test[i, :, :, 0] - y_pred[i, :, :, 0]
    morans_i = compute_morans_i(residuals)
    morans_i_values.append(morans_i)
morans_i_mean = np.mean(morans_i_values)

# === Affichage des résultats ===
print(f"✅  RMSE = {rmse:.5f}")
print(f"✅  MAE  = {mae:.5f}")
print(f"✅  NSE  = {nse:.5f}")
print(f"✅  SSIM = {ssim_mean:.5f} (moyenne sur {len(ssim_values)} échantillons)")
print(f"✅  Moran's I = {morans_i_mean:.5f} (moyenne sur {len(morans_i_values)} échantillons)")

# === Sauvegarde des métriques ===
with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"MAE : {mae:.6f}\n")
    f.write(f"NSE : {nse:.6f}\n")
    f.write(f"SSIM: {ssim_mean:.6f}\n")
    f.write(f"Moran's I: {morans_i_mean:.6f}\n")
    f.write(f"--- Détails SSIM ---\n")
    for i, val in enumerate(ssim_values):
        f.write(f"  Échantillon {i}: {val:.6f}\n")
    f.write(f"--- Détails Moran's I ---\n")
    for i, val in enumerate(morans_i_values):
        f.write(f"  Échantillon {i}: {val:.6f}\n")

# === Export des images concaténées (prédiction + réalité) ===
seuil_mer = 0.1
cmap = cm.get_cmap("RdYlGn")

for idx in range(len(y_pred)):
    # Image prédite (avec seuil)
    pred = y_pred[idx, :, :, 0]
    pred_rgba = cmap(pred)[:, :, :3]
    pred_rgba[pred < seuil_mer] = [1, 0, 0]
    pred_rgb = (pred_rgba * 255).astype(np.uint8)

    # Image réelle (sans seuil)
    real = y_test[idx, :, :, 0]
    real_rgba = cmap(real)[:, :, :3]
    real_rgb = (real_rgba * 255).astype(np.uint8)

    # Concaténation horizontalement
    concat = np.concatenate([pred_rgb, real_rgb], axis=1)  # (H, 2*W, 3)

    # Affichage avec titres
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(concat)
    ax.axis("off")
    ax.set_title(f"Prédiction (gauche) vs Réel (droite) — #{idx}")
    plt.tight_layout()

    # Sauvegarde image
    save_path = os.path.join(OUT_DIR, f"comparison_{idx:04d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

print(f"💾  {len(y_pred)} comparaisons PNG sauvegardées dans {OUT_DIR}")