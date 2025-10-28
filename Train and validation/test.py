from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

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

print(f"✅  RMSE = {rmse:.5f}")
print(f"✅  MAE  = {mae:.5f}")
print(f"✅  NSE  = {nse:.5f}")

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"RMSE: {rmse:.6f}\nMAE : {mae:.6f}\nNSE : {nse:.6f}\n")

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
