import numpy as np
import pandas as pd
import geopy.distance as distance
from datetime import timedelta



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
    