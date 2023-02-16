import numpy as np
import pandas as pd
from pathlib import Path
import odc.stac
import pystac_client
import planetary_computer
import rioxarray
import geopandas as gpd
import geopy.distance as distance
from shapely.geometry import Point
import xarray
import xrspatial
from datashader.transfer_functions import shade, stack
from datashader.colors import Elevation
import matplotlib.pyplot as plt
from rasterio import plot
import geopandas as gpd
from shapely.geometry import box,Point
import cv2
from datetime import timedelta
from tqdm import tqdm




#-----------------------------------------------------------------------------------------------------
#giving the latitude and longitude coordinates of a location and returns location_url on google map
def get_location_url_google_maps(latitude, longitude):
    return "https://www.google.com/maps/search/?api=1&query=" + str(latitude) + "," + str(longitude)

#-----------------------------------------------------------------------------------------------------
# get our bounding box to search latitude and longitude coordinates
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



#-----------------------------------------------------------------------------------------------------
def normalize_elevation(arr):
    ''' Function to scale an input array to [0.0, 1.0] '''
    arr_min = -100 # lowest point: death valley
    arr_max = 4500 # highest point
    
    arr_range = arr_max - arr_min
    scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')
    
    # limit values to 1.0
    scaled[scaled >= 1.0] = 1.0
    arr_new = ( (scaled)) 
    return arr_new

#-----------------------------------------------------------------------------------------------------
def crop_copernicus_image(item, bounding_box):
    """
    Given a STAC item from copernicus and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.
    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = rioxarray.open_rasterio(planetary_computer.sign(item.assets["data"].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )

    return image.to_numpy()

#-----------------------------------------------------------------------------------------------------

def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%dT"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"

    return date_range

#-----------------------------------------------------------------------------------------------------

def collect_elevation_layers(row, copernicus_catalog, image_size=128, verbose=False):
    # Search in catalog based on bounding box and date
    # calculate rough estimate of the bounding box size based on copernicus resolution 10m/pixel
    meter_buffer = ((10 * image_size) / 2) - 5
    bbox = get_bounding_box(row.latitude, row.longitude, meter_buffer=meter_buffer)

    date_range = get_date_range(row.date, time_buffer_days=15)

    # search the planetary computer copernicus collections
    search_results = copernicus_catalog.search(
        collections=["cop-dem-glo-30"],
    intersects={"type": "Point", "coordinates":  [row.longitude,row.latitude, ]},
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

    # check how many images are available
    num_copernicus = len(item_details[item_details.platform.str.contains("TanDEM-X")])
    if verbose:
        print(f"Datapoint ID: {row.uid}")
        print(f"Number of copernicus items found: {num_copernicus}")

    # collect layer from catalog (based on copernicus )
   
    if num_copernicus > 0:
        # If is available, it is first choice.
        if verbose: 
            print('TanDEM-X')
        # Select image by date.
        item = (
            item_details[item_details.platform.str.contains("TanDEM-X")]
            .sort_values(by="datetime", ascending=False)
            .iloc[0]
            ).item_obj
        # collect layers in dict
        
        image = rioxarray.open_rasterio(planetary_computer.sign(item.assets["data"].href)).rio.clip_box(
        minx=bbox[0],
        miny=bbox[1],
        maxx=bbox[2],
        maxy=bbox[3],
        crs="EPSG:4326",
    )
        # normalize pixel values into range 0.0 to 1.0
        normalized = normalize_elevation(image[0])
        # resize image if it does not match "image_size"
        if (normalized.shape[0] != image_size) or (normalized.shape[1] != image_size):
            #print(f'Shape {name}: {normalized.shape} -->RESIZED!!!')
            elevation_layer= cv2.resize(normalized, ( image_size, image_size), interpolation = cv2.INTER_AREA)
        else:
            elevation_layer = normalized
    else:
        # use array of zeros when no image is found
        if verbose:
            print("No Image was found!")
        elevation_layer = np.zeros([image_size, image_size])
    
    return elevation_layer



#-----------------------------------------------------------------------------------------------------

# function to calculate elevation features
def get_elevation_features(elevation_grid):
    elev_mean   = np.mean(elevation_grid)
    elev_median = np.median(elevation_grid)
    elev_min    = np.min(elevation_grid)
    elev_max    = np.max(elevation_grid)
    features = {'elev_mean':   elev_mean,
                'elev_median': elev_median,
                'elev_min':    elev_min, 
                'elev_max':    elev_max}
    return features

#-----------------------------------------------------------------------------------------------------