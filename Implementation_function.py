import os
import numpy as np
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def predict_ndvi_sequence(folder_path, model_path, output_path="pred_result"):
    os.makedirs(output_path, exist_ok=True)

    # === load 12 images NDVI .tif ===
    tif_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith(".tif")
    ])
    assert len(tif_paths) == 12, "❌ no enough imah=ges in the directory"

    def load_ndvi(path):
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
            img = np.nan_to_num(img, nan=0.0)
            img = (img - (-0.2)) / (0.5)  # Normalisation NDVI 
            return np.clip(img, 0, 1)     



    # === Separatevthe 9 images of input from the target image ===
    images_input = [load_ndvi(p) for p in tif_paths[:12]]      # (12, H, W)
    image_real   = load_ndvi2(tif_paths[9])                    # (H, W)

    sequence = np.stack(images_input, axis=0)[..., np.newaxis]   # (12, H, W, 1)
    sequence = np.expand_dims(sequence, axis=0)                  # (1, 12, H, W, 1)

    # === Prediction ===
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(sequence, verbose=0)[0, :, :, 0]       # (H, W)

   

    # === Visualisation PNG avec palette RdYlGn ===
    cmap = cm.get_cmap("RdYlGn")

    def save_colored_ndvi(ndvi_array, filename):
        colored = cmap(ndvi_array)[:, :, :3]            # RGB only
        colored = (colored * 255).astype(np.uint8)
        plt.imsave(os.path.join(output_path, filename), colored)

    save_colored_ndvi(y_pred, "pred_ndvi.png")
    save_colored_ndvi(image_real, "real_ndvi.png")

    print("✅ image sauvgarded in:", output_path)

# === example of call
predict_ndvi_sequence(
    folder_path="inference_seq",       
    model_path="model/model.keras",
    output_path="results/predicted_ndvi"
)
