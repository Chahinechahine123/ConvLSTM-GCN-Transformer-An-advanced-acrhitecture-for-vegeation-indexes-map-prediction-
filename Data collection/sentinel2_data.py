import ee
import os
import urllib.request

# Authenticate and initialize
ee.Authenticate(auth_mode='localhost')
ee.Initialize(project=' Your cloud repo')

# Define region and time range
My_region = ee.Geometry.Rectangle([9.7, 37.2, 9.9, 37.3])
start_year, end_year = 2016, 2025
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# Load Sentinel-2 SR collection
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(My_region) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))

# Cloud mask using QA60
def mask_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# NDVI computation
def compute_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi.copyProperties(image, image.propertyNames())

# Prepare NDVI collection
s2_ndvi = s2.map(mask_clouds).map(compute_ndvi)

# Monthly composite and export
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        img = s2_ndvi.filterDate(start, end).mean()
        file_name = f"S2_NDVI_{year}_{month:02d}.tif"
        path = os.path.join(output_dir, file_name)
        print(f"Downloading {file_name} ...")

        url = img.clip(My_region).getDownloadURL({
            'scale': 10,
            'region': My_region,
            'format': 'GEO_TIFF'
        })
        urllib.request.urlretrieve(url, path)
        print(f"Saved → {path}")
