from datetime import date, timedelta, datetime
import pandas as pd
import xarray as xr
import requests
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import matplotlib.pyplot as plt
import cmocean
import tempfile
import numpy as np

# Not used directly, but used via xarray
#import cfgrib #you need to install some dependencies: https://pypi.org/project/cfgrib/0.8.4.5/ and also look here https://github.com/ecmwf/eccodes-python/issues/54

def join_time_values_three(a, b, c):
    result = [0] * (len(a) + len(b) + len(c))
    result[::3]  = a[::-1]
    result[1::3] = b[::-1]
    result[2::3] = c[::-1]
    return result

def join_time_values(a, b, c, d):
    result = [0] * (len(a) + len(b) + len(c) + len(d))
    result[::4]  = a[::-1]
    result[1::4] = b[::-1]
    result[2::4] = c[::-1]
    result[3::4] = d[::-1]
    return result

def replace_nan(x):
    if x=="nan":
        return np.nan
    else :
        return float(x)

def convert_to_str(data, features):
    for feature in features : 
        data[feature]=data[feature].apply([str])
    return data

def convert_str_to_list(data, features):
    for feature in features : 
        data[feature]=data[feature].apply(lambda x: [ replace_nan(X) for X in x.strip('[]').split(",")])
    return data


def get_ds():
    """get and load grip2 data for the NOAA HRRR model"""
    blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
    sector = "conus"
    yesterday = date.today() - timedelta(days=1)
    cycle = 17 
    forecast_hour = 1   # offset from cycle time
    product = "wrfsfcf" # 2D surface levels
    # Put it all together
    file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
    url = f"{blob_container}/hrrr.{yesterday:%Y%m%d}/{sector}/{file_path}"
    
    r = requests.get(f"{url}.idx")
    idx = r.text.splitlines()
    
    sfc_temp_idx = [l for l in idx if ":TMP:surface" in l][0].split(":")
    # Pluck the byte offset from this line, plus the beginning offset of the next line
    line_num = int(sfc_temp_idx[0])
    range_start = sfc_temp_idx[1]
    # The line number values are 1-indexed, so we don't need to increment it to get the next list index,
    # but check we're not already reading the last line
    next_line = idx[line_num].split(':') if line_num < len(idx) else None
    # Pluck the start of the next byte offset, or nothing if we were on the last line
    range_end = next_line[1] if next_line else None

    file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)
    headers = {"Range": f"bytes={range_start}-{range_end}"}
    resp = requests.get(url, headers=headers, stream=True)
    with file as f:
        f.write(resp.content)
    ds = xr.open_dataset(file.name, engine='cfgrib', 
                     backend_kwargs={'indexpath':''})
    return ds


# def get_ds_aws_tarray(day_date, cycle):
#     """get and load grip2 data for the NOAA HRRR model from aws"""
#     sector = "conus"
#     yesterday = day_date
#     cycle = cycle 
#     forecast_hour = 1   # offset from cycle time
#     product = "wrfsfcf" # 2D surface levels
#     # Put it all together
#     file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
#     url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{yesterday:%Y%m%d}/{sector}/{file_path}"

#     r = requests.get(f"{url}.idx")
#     idx = r.text.splitlines()
    
#     try: 
#         sfc_temp_idx = [l for l in idx if ":TMP:surface" in l][0].split(":")
#         # Pluck the byte offset from this line, plus the beginning offset of the next line
#         line_num = int(sfc_temp_idx[0])
#         range_start = sfc_temp_idx[1]
#         # The line number values are 1-indexed, so we don't need to increment it to get the next list index,
#         # but check we're not already reading the last line
#         next_line = idx[line_num].split(':') if line_num < len(idx) else None
#         # Pluck the start of the next byte offset, or nothing if we were on the last line
#         range_end = next_line[1] if next_line else None

#         file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)
#         headers = {"Range": f"bytes={range_start}-{range_end}"}
#         resp = requests.get(url, headers=headers, stream=True)
#         with file as f:
#             f.write(resp.content)
#         ds = xr.open_dataset(file.name, engine='cfgrib', 
#                          backend_kwargs={'indexpath':''})
#         return ds.t.values, False #war das komma wichtig????
#     except:
#         return 1, True




def get_ds_aws_array(day_date, cycle, param_layer, forecast_param):
    """get and load grip2 data for the NOAA HRRR model from aws, 
    returns an array with just the specified parameter"""
    
    param_layer = param_layer
    forecast_param = forecast_param
    sector = "conus"
    yesterday = day_date
    cycle = cycle 
    forecast_hour = 1   # offset from cycle time
    product = "wrfsfcf" # 2D surface levels
    # Put it all together
    file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
    url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{yesterday:%Y%m%d}/{sector}/{file_path}"

    r = requests.get(f"{url}.idx")
    idx = r.text.splitlines()

    try: 
        sfc_temp_idx = [l for l in idx if param_layer in l][0].split(":")
        # Pluck the byte offset from this line, plus the beginning offset of the next line
        line_num = int(sfc_temp_idx[0])
        range_start = sfc_temp_idx[1]
        # The line number values are 1-indexed, so we don't need to increment it to get the next list index,
        # but check we're not already reading the last line
        next_line = idx[line_num].split(':') if line_num < len(idx) else None
        # Pluck the start of the next byte offset, or nothing if we were on the last line
        range_end = next_line[1] if next_line else None

        file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)
        headers = {"Range": f"bytes={range_start}-{range_end}"}
        resp = requests.get(url, headers=headers, stream=True)
        with file as f:
            f.write(resp.content)
        ds = xr.open_dataset(file.name, engine='cfgrib', 
                         backend_kwargs={'indexpath':''})
        return ds[forecast_param].values, False 
    except:
        return 1, True





#convert dates into the right format
def get_start_date(start_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    return start_date


#make function that computes the gridpoints for a given coordinates
def get_grids(stn_lon, stn_lat, lons, lats):
    abslat = np.abs(lats-stn_lat)
    abslon= np.abs(lons-stn_lon)
    c = np.maximum(abslon, abslat)
    x, y = np.where(c==c.min())
    x = x[0]
    y = y[0]
    return x, y

# save gridpoints in metadatafile
def save_grids(metadata, ds):
    lons = ds.longitude.values
    lats = ds.latitude.values
    for i in range(len(metadata.uid)):
        stn_lon = metadata.longitude_trans[i]
        stn_lat = metadata.latitude[i]
        x, y = get_grids(stn_lon, stn_lat, lons, lats)
        metadata.x_grid.loc[i] = x
        metadata.y_grid.loc[i] = y
    return metadata