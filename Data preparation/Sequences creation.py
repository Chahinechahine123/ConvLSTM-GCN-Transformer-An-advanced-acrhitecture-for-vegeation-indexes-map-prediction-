import numpy as np
import glob
import rasterio
import os
import matplotlib.pyplot as plt

# === Dossier d'entrée (images MODIS) ===
data_dir = "images/modis_clipped"
seq_len = 9

# === Chargement des fichiers .tif (float uniquement) ===
files = sorted(glob.glob(os.path.join(data_dir, "*_float.tif")))
print("✅", len(files), "images NDVI (float) trouvées.")
if len(files) <= seq_len:
    raise RuntimeError("❌ Pas assez d'images pour créer des séquences de test.")

# === Chargement des NDVI MODIS (normalisé [-1,1] → [0,1]) ===
def read_ndvi(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0, 1)

stack = np.stack([read_ndvi(p) for p in files])  # (N, H, W)
H, W = stack.shape[1:]

# === Génération des séquences temporelles ===
X_test, y_test = [], []
for i in range(len(stack) - seq_len):
    X_test.append(stack[i:i+seq_len])     # (9, H, W)
    y_test.append(stack[i+seq_len])       # (H, W)

X_test = np.array(X_test)[..., np.newaxis]   # (N, 9, H, W, 1)
y_test = np.array(y_test)[..., np.newaxis]   # (N, H, W, 1)

print("✅ Données MODIS préparées pour test :")
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# === Sauvegarde ===
out_dir = "brute_data/modis"
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "X_test.npy"), X_test)
np.save(os.path.join(out_dir, "y_test.npy"), y_test)

# === Aperçu d’un exemple ===
plt.figure(figsize=(8, 3.5))
plt.subplot(1, 2, 1)
plt.imshow(X_test[0, -1, :, :, 0], cmap="RdYlGn")
plt.title("Dernière entrée (t-1)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(y_test[0, :, :, 0], cmap="RdYlGn")
plt.title("Image cible (t)")
plt.axis("off")

plt.tight_layout()
plt.show()

