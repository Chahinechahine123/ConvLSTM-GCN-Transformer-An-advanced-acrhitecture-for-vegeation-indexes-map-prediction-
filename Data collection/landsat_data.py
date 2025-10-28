import ee
import os
import urllib.request

# Authenticate and initialize Earth Engine
ee.Authenticate(auth_mode='localhost')
ee.Initialize(project='Your cloud repo')

# Define region and time range
My_region = ee.Geometry.Rectangle([9.7, 37.2, 9.9, 37.3])
start_year, end_year = 1996, 2025
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# Load Landsat SR collections
l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
landsat = l5.merge(l7).merge(l8).merge(l9)

# Cloud mask
def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)

# Compute NDVI for any Landsat version
def compute_ndvi(image):
    bands = image.bandNames().getInfo()
    if 'SR_B5' in bands:  # Landsat 8/9
        nir, red = image.select('SR_B5'), image.select('SR_B4')
    else:  # Landsat 5/7
        nir, red = image.select('SR_B4'), image.select('SR_B3')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return ndvi.copyProperties(image, image.propertyNames())

# Prepare NDVI collection
landsat_ndvi = landsat.filterBounds(My_region).map(mask_clouds).map(compute_ndvi)

# Monthly composite and export
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        img = landsat_ndvi.filterDate(start, end).mean()
        file_name = f"Landsat_NDVI_{year}_{month:02d}.tif"
        path = os.path.join(output_dir, file_name)
        print(f"Downloading {file_name} ...")

        url = img.clip(My_region).getDownloadURL({
            'scale': 30,
            'region': My_region,
            'format': 'GEO_TIFF'
        })
        urllib.request.urlretrieve(url, path)
        print(f"Saved → {path}")
