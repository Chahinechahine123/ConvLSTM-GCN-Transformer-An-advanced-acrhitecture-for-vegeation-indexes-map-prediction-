import rasterio
import numpy as np
import os
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter
from tqdm import tqdm
from glob import glob

def load_ndvi_series(input_dir, invalid_threshold=0.0):
    files = sorted(glob(os.path.join(input_dir, "*.tif")))
    series = []
    profile = None

    if len(files) == 0:
        raise RuntimeError("No TIFF files found in input directory")

    for idx, f in enumerate(files):
        with rasterio.open(f) as src:
            img = src.read(1).astype(np.float32)

            # Save profile once (important!)
            if idx == 0:
                profile = src.profile.copy()

            # DEBUG (optional – you can remove later)
            print(os.path.basename(f),
                  "Min:", np.nanmin(img),
                  "Max:", np.nanmax(img))

            # Handle nodata
            if src.nodata is not None:
                img[img == src.nodata] = np.nan

            # 🔴 CRITICAL FIX:
            # NDVI <= 0 → treated as missing (clouds, shadows, water)
            img[img <= invalid_threshold] = np.nan

            series.append(img)

    return np.stack(series), files, profile



def save_series(series, files, out_dir, profile):
    os.makedirs(out_dir, exist_ok=True)

    profile_out = profile.copy()
    profile_out.update(dtype=rasterio.float32, nodata=np.nan)

    for img, filepath in zip(series, files):
        name = os.path.basename(filepath)  # <-- change here
        with rasterio.open(
            os.path.join(out_dir, name),
            "w",
            **profile_out
        ) as dst:
            dst.write(img.astype(np.float32), 1)

def hybrid_gap_filling(series):
    filled = series.copy()

    # 1️⃣ Monthly maximum (safe)
    monthly_max = np.nanmax(
        np.where(np.isnan(filled), -np.inf, filled), axis=0
    )
    monthly_max[~np.isfinite(monthly_max)] = np.nan

    for t in range(filled.shape[0]):
        mask = np.isnan(filled[t])
        filled[t][mask] = monthly_max[mask]

    # 2️⃣ Temporal smoothing (t-1, t, t+1)
    for t in range(1, filled.shape[0] - 1):
        mean_3 = np.nanmean(filled[t-1:t+2], axis=0)
        mask = np.isnan(filled[t]) & ~np.isnan(mean_3)
        filled[t][mask] = mean_3[mask]

    # 3️⃣ Spatial 3×3 mean
    for t in range(filled.shape[0]):
        spatial_mean = uniform_filter(
            np.nan_to_num(filled[t], nan=0.0), size=3
        )
        valid_count = uniform_filter(
            (~np.isnan(filled[t])).astype(np.float32), size=3
        )
        spatial_mean = np.where(valid_count > 0, spatial_mean / valid_count, np.nan)

        mask = np.isnan(filled[t])
        filled[t][mask] = spatial_mean[mask]

    # 4️⃣ Global fallback
    global_mean = np.nanmean(filled)
    filled[np.isnan(filled)] = global_mean

    return filled
def linear_interpolation(series):
    filled = series.copy()
    T, H, W = filled.shape

    for i in range(H):
        for j in range(W):
            ts = filled[:, i, j]
            valid = ~np.isnan(ts)
            if valid.sum() >= 2:
                filled[:, i, j] = np.interp(
                    np.arange(T), np.where(valid)[0], ts[valid]
                )
    return filled
def savgol_gap_filling(series, window=5, poly=2):
    filled = series.copy()
    T, H, W = filled.shape

    for i in range(H):
        for j in range(W):
            ts = filled[:, i, j]
            valid = ~np.isnan(ts)
            if valid.sum() >= window:
                interp = np.interp(np.arange(T), np.where(valid)[0], ts[valid])
                filled[:, i, j] = savgol_filter(interp, window, poly)
    return filled
def spatiotemporal_mean(series):
    filled = series.copy()
    T = filled.shape[0]

    for t in range(1, T - 1):
        cube = np.nanmean(filled[t-1:t+2], axis=0)

        spatial = uniform_filter(
            np.nan_to_num(cube, nan=0.0), size=3
        )
        count = uniform_filter(
            (~np.isnan(cube)).astype(np.float32), size=3
        )
        spatial = np.where(count > 0, spatial / count, np.nan)

        mask = np.isnan(filled[t])
        filled[t][mask] = spatial[mask]

    filled[np.isnan(filled)] = np.nanmean(filled)
    return filled
input_dir = "in"
output_dir = "output"

series, files, profile = load_ndvi_series(input_dir)

methods = {
    "hybrid": hybrid_gap_filling,
    "linear_interp": linear_interpolation,
    "savgol": savgol_gap_filling,
    "spatiotemporal_mean": spatiotemporal_mean
}

for name, func in methods.items():
    print(f"Running {name}...")
    filled = func(series)
    save_series(filled, files, os.path.join(output_dir, name), profile)
