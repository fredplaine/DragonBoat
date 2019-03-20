# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:13:43 2019

@author: Freddie
"""

import pandas as pd
import gpxpy
import matplotlib.pyplot as plt
from rdp import rdp
import numpy as np
from pykalman import KalmanFilter
import math

EARTH_RADIUS = 6371*1000 #6378.137 * 1000

def haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2):
    """
    Haversine distance between two points, expressed in meters.
    Implemented from http://www.movable-type.co.uk/scripts/latlong.html
    """
    d_lat = (latitude_1 - latitude_2)
    d_lon = (longitude_1 - longitude_2)
    lat1 = (latitude_1)
    lat2 = (latitude_2)

    a = math.sin(d_lat/2) * math.sin(d_lat/2) + \
        math.sin(d_lon/2) * math.sin(d_lon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = EARTH_RADIUS * c

    return d

with open('GPSdata.2019.03.19.gpx') as fh:
    file = gpxpy.parse(fh)

#file.remove_elevation() 
    
segment =file.tracks[0].segments[0]


for i in range (1,len(segment.points)):
    lat1 = math.radians(segment.points[i-1].latitude)
    lat2 = math.radians(segment.points[i].latitude)
    lon1 = math.radians(segment.points[i-1].longitude)
    lon2 = math.radians(segment.points[i].longitude)
    segment.points[i].speed = haversine_distance(lat1, lon1, lat2, lon2) * 3.6
    

coords = pd.DataFrame ([
        {'lat': p.latitude,
         'lon': p.longitude,
         'ele': p.elevation,
         'speed': p.speed,
         'time': p.time.replace(tzinfo=None)} for p in segment.points])
coords.set_index('time', drop=True, inplace=True)

coords = coords.resample('1S').asfreq()

measurements = np.ma.masked_invalid(coords[['lon','lat','ele']].values)
plt.plot(measurements[:,0], measurements[:,1])
filled_coords = coords.fillna(method='pad').loc[coords.ele.isnull()]
plt.plot(filled_coords['lon'].values,filled_coords['lat'].values,'ro')

segment.points[0].speed, segment.points[-1].speed= 0.,0.
file.add_missing_speeds()
speed = np.array([p.speed for p in segment.points])
plt.plot(speed)
