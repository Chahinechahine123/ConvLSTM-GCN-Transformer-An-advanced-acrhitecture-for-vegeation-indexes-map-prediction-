import ee
import os

# Authenticate and initialize Earth Engine
ee.Authenticate(auth_mode='localhost')
ee.Initialize(project='Your cloud repo')

# Define study region (Bizerte, Tunisia)
My_region = ee.Geometry.Rectangle([9.7, 37.2, 9.9, 37.3])

# Define time range
start_year, end_year = 2000, 2025

# Local output folder
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# Load MODIS Surface Reflectance collection
modis_sr = ee.ImageCollection('MODIS/006/MOD09GA').filterBounds(My_region)

# Function to compute NDVI
def compute_ndvi(image):
    ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('NDVI')
    return ndvi.copyProperties(image, image.propertyNames())

# Compute NDVI collection
modis_ndvi = modis_sr.map(compute_ndvi)

# Function to generate monthly mean composite
def monthly_composite(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')
    return modis_ndvi.filterDate(start, end).mean().set('year', year, 'month', month)

# Export monthly NDVI to local drive
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        img = monthly_composite(year, month)
        file_name = f"MODIS_NDVI_{year}_{month:02d}.tif"
        path = os.path.join(output_dir, file_name)
        print(f"Downloading {file_name} ...")

        # Download as GeoTIFF
        url = img.clip(My_region).getDownloadURL({
            'scale': 250,
            'region': My_region,
            'format': 'GEO_TIFF'
        })
        import urllib.request
        urllib.request.urlretrieve(url, path)
        print(f"Saved → {path}")
