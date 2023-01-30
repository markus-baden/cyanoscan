import cv2
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import geopy.distance as distance
import geopandas as gpd
from shapely.geometry import Point

import rioxarray
from IPython.display import Image
from PIL import Image as PILImage

import planetary_computer as pc
from pystac_client import Client

SENTINEL_BANDS_DESCR = {
    #'AOT': 'Aerosol optical thickness (AOT)',
    #'B01': 'Coastal aerosol - 60m',
    'B04': 'Red - 10m',
    'B03': 'Green - 10m',
    'B02': 'Blue - 10m',
    #'B05': 'Vegetation red edge 1 - 20m',
    #'B06': 'Vegetation red edge 2 - 20m',
    #'B07': 'Vegetation red edge 3 - 20m',
    'B08': 'NIR - 10m',
    #'B09': 'Water vapor - 60m',
    #'B11': 'Short Wave IR (1.6) - 20m',
    #'B12': 'Short Wave IR (2.2) - 20m',
    #'B8A': 'Vegetation red edge 4 - 20m',
    #'SCL': 'Scene classfication map (SCL)',
    #'WVP': 'Water vapour (WVP)',
}
SENTINEL_BANDS = list(SENTINEL_BANDS_DESCR.keys())

LANDSAT_BANDS_DESCR = {
    #'qa': 'Surface Temperature Quality Assessment Band',
    #'ang': 'Angle Coefficients File',
    'red': 'Red Band',
    'green': 'Green Band',
    'blue': 'Blue Band',
    #'drad': 'Downwelled Radiance Band',
    #'emis': 'Emissivity Band',
    #'emsd': 'Emissivity Standard Deviation Band',
    #'trad': 'Thermal Radiance Band',
    #'urad': 'Upwelled Radiance Band',
    #'atran': 'Atmospheric Transmittance Band',
    #'cdist': 'Cloud Distance Band',
    'nir08': 'Near Infrared Band 0.8',
    #'lwir11': 'Surface Temperature Band',
    #'swir16': 'Short-wave Infrared Band 1.6',
    #'swir22': 'Short-wave Infrared Band 2.2',
    #'coastal': 'Coastal/Aerosol Band',
    #'mtl.txt': 'Product Metadata File (txt)',
    #'mtl.xml': 'Product Metadata File (xml)',
    #'mtl.json': 'Product Metadata File (json)',
    #'qa_pixel': 'Pixel Quality Assessment Band',
    #'qa_radsat': 'Radiometric Saturation and Terrain Occlusion Quality Assessment Band',
    #'qa_aerosol': 'Aerosol Quality Assessment Band',
    #'tilejson': 'TileJSON with default rendering',
    #'rendered_preview': 'Rendered preview',
}
LANDSAT_BANDS = list(LANDSAT_BANDS_DESCR.keys())

BAND_NAMES = ['red', 'green', 'blue', 'nir']


def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%dT"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"

    return date_range

def get_bounding_box(latitude, longitude, meter_buffer=50000):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meter_buffer)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination((latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination((latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination((latitude, longitude), bearing=90)[1]

    return [min_long, min_lat, max_long, max_lat]

def normalize_landsat(arr):
    ''' Function to scale an input array to [0, 255] '''
    #arr_min = arr.min()
    arr_min = 0 #arr.min()
    
    # for Sentinel L2A the max value used for creating the visual image channel is 2000
    #arr_max = arr.max()
    arr_max = 10000 #arr.max()
    
    # Check the original min and max values
    #print('Min: %.3f, Max: %.3f' % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')

    # limit values to 1.0
    scaled[scaled >= 1.0] = 1.0
    arr_new = ( (scaled ))#* 255.0) )
    
    return arr_new#.astype(int)

def normalize_sentinel(arr):
    ''' Function to scale an input array to [0, 255] '''
    #arr_min = arr.min()
    arr_min = 0 #arr.min()
    
    # for Sentinel L2A the max value used for creating the visual image channel is 2000
    #arr_max = arr.max()
    arr_max = 2000 #arr.max()
    
    # Check the original min and max values
    #print('Min: %.3f, Max: %.3f' % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')
    
    # limit values to 1.0
    scaled[scaled >= 1.0] = 1.0
    arr_new = ( (scaled)) #* 255.0) )
    
    return arr_new#.astype(int)



def collect_sat_image_layers(row, sat_image_catalog, image_size=128, verbose=False):
    # Search in catalog based on bounding box and date
    # calculate rough estimate of the bounding box size based on Sentinel resolution 10m/pixel
    meter_buffer = ((10 * image_size) / 2) - 5
    bbox = get_bounding_box(row.latitude, row.longitude, meter_buffer=meter_buffer)

    date_range = get_date_range(row.date, time_buffer_days=15)

    # search the planetary computer sentinel-l2a and landsat level-2 collections
    search_results = sat_image_catalog.search(
        collections=["sentinel-2-l2a", "landsat-c2-l2"], bbox=bbox, datetime=date_range
    )

    # get details of all of the items returned
    items = [item for item in search_results.get_all_items()]
    item_details = pd.DataFrame([
        {
            "datetime": item.datetime.strftime("%Y-%m-%d"),
            "platform": item.properties["platform"],
            "min_long": item.bbox[0],
            "max_long": item.bbox[2],
            "min_lat": item.bbox[1],
            "max_lat": item.bbox[3],
            "bbox": item.bbox,
            "cloud_cov": item.properties['eo:cloud_cover'],
            "item_obj": item,
        }
        for item in items])

    # check which rows actually contain the sample location
    item_details["contains_sample_point"] = (
    (item_details.min_lat < row.latitude)
    & (item_details.max_lat > row.latitude)
    & (item_details.min_long < row.longitude)
    & (item_details.max_long > row.longitude)
    )
    item_details = item_details[item_details["contains_sample_point"]]

    # check how many Sentinel and Landsat images are available
    num_sentinel = len(item_details[item_details.platform.str.contains("Sentinel")])
    num_landsat  = len(item_details[item_details.platform.str.contains("landsat")])
    if verbose:
        print(f"Datapoint ID: {row.uid}")
        print(f"Number of Sentinel items found: {num_sentinel}")
        print(f"Number of Landsat items found: {num_landsat}")

    # collect layers from catalog (based on Sentinel or Landsat or none)
    image_layers = dict.fromkeys(BAND_NAMES, np.array(0))
    if num_sentinel > 0:
        # If sentinel is available, it is first choice.
        if verbose: 
            print('Sentinel')
        # Select closest image by date.
        # TODO: Refine selection by using the less cloudy image when there's more than one image for the date??
        item = (
            item_details[item_details.platform.str.contains("Sentinel")]
            .sort_values(by="datetime", ascending=False)
            .iloc[0]
            ).item_obj
        # collect layers in dict
        for band, name in zip(SENTINEL_BANDS, BAND_NAMES):
            image = rioxarray.open_rasterio(pc.sign(item.assets[band].href)).rio.clip_box(
                minx=bbox[0],
                miny=bbox[1],
                maxx=bbox[2],
                maxy=bbox[3],
                crs="EPSG:4326",
            ).to_numpy()
            # normalize pixel values into range 0.0 to 1.0
            normalized = normalize_sentinel(image[0])
            # resize image if it does not match "image_size"
            if (normalized.shape[0] != image_size) or (normalized.shape[1] != image_size):
                #print(f'Shape {name}: {normalized.shape} -->RESIZED!!!')
                image_layers[name] = cv2.resize(normalized, ( image_size, image_size), interpolation = cv2.INTER_AREA)
            else:
                image_layers[name] = normalized

    elif num_landsat > 0:
        # If only Landsat is available we use it instead
        if verbose:
            print('Landsat')
        item = (
            item_details[item_details.platform.str.contains("landsat")]
            .sort_values(by="datetime", ascending=False)
            .iloc[0]
            ).item_obj
        # collect layers in dict
        for band, name in zip(LANDSAT_BANDS, BAND_NAMES):
            image = odc.stac.stac_load([pc.sign(item)], bands=[band], bbox=bbox).isel(time=0)
            # scale pixel values into practical range
            normalized = normalize_landsat(image[[band]].to_array().to_numpy()[0])
            # upscale landsat images to same pixel size as sentinel
            image_layers[name] = cv2.resize(normalized, ( image_size, image_size), interpolation = cv2.INTER_AREA)
    else:
        # use array of zeros when no image is found
        if verbose:
            print("No Image was found!")
        for band, name in zip(LANDSAT_BANDS, BAND_NAMES):
            image_layers[name] = np.zeros([image_size, image_size])
    
    return image_layers
    


def get_ndwi_mask(image_layers, ndwi_limit=0.12):
    """Calculates image mask based on NDWI
    NDWI - Normalized Difference Water Index
    NDWI = (green - NIR) / (green + NIR)
    source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
    another source: https://en.wikipedia.org/wiki/Normalized_difference_water_index

    Args:
        image_layers (_type_): Satellite image layers
        ndwi_limit (float, optional): Limit value for masking. According to sources 0.3 is standard, 
            but achieved better results with defaulting to 0.12.

    Returns:
        ndwi_mask: True/False mask, True where water is found
        ndwi_image: 1-layer image of the NDWI
    """
    # Calculate NDWI layers: (green - NIR) / (green + NIR)
    a = (image_layers['green'] - image_layers['nir']) 
    b = (image_layers['green'] + image_layers['nir'])
    # this line is safe against division by 0 errors:
    ndwi_image = np.array([np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b!=0)]) 
    
    # create mask based on NDWI values
    ndwi_mask = (ndwi_image[0] > ndwi_limit)
    return ndwi_mask, ndwi_image