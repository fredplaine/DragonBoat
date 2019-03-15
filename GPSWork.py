# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:56:41 2019

@author: Freddie

Based on  https://www.youtube.com/watch?v=9Q8nEA_0ccg&list=WL&index=46&t=0s
"""

import pandas as pd
import gpxpy
import matplotlib.pyplot as plt
from rdp import rdp
import numpy as np
from pykalman import KalmanFilter




with open('GPS_HV1.xml') as fh:
    gpx_file = gpxpy.parse(fh)
    
#print(gpx_file.get_uphill_downhill())

#segment =gpx_file.tracks[0].segments[0]
#coords = pd.DataFrame ([
#        {'lat': p.latitude,
#         'lon': p.longitude,
#         'ele': p.elevation,
#         'speed': p.speed,
#         'time': p.time} for p in segment.points])
#coords.set_index('time', drop=True, inplace=True)
#print(coords.head(3))

segment =gpx_file.tracks[0].segments[0]
coords = pd.DataFrame ([{'idx': i,
                         'lat': p.latitude,
                         'lon': p.longitude,
                         'ele': p.elevation,
                         'speed': p.speed,
                         'time': p.time.replace(tzinfo=None)} for i,p in enumerate(segment.points)])
coords.set_index('time', inplace=True)
#print(coords.head(2))

#for idx in coords:
#    =ACOS(SIN(H2/180)*SIN(H3/180)+COS(H2/180)*COS(H3/180)*COS(I2/180-I3/180))
#    =ACOS(SIN(lat-1/180)*SIN(lat/180)+COS(lat-1/180)*COS(lat/180)*COS(lon-1/180-lon/180))
#    print (idx)
    #lat2 = 
    #lon1 =
    #lon2 =
    #coords['speed'] = coords['speed'][idx-1]
    
coords ['ele'] = 0

#plt.plot(np.diff(coords.index))

coords = coords.resample('1S').asfreq()
#print(coords.loc[coords.ele.isnull()].head())

measurements = np.ma.masked_invalid(coords[['lon','lat','ele']].values)
plt.plot(measurements[:,0], measurements[:,1])
filled_coords = coords.fillna(method='pad').ix[coords.ele.isnull()]
plt.plot(filled_coords['lon'].values,filled_coords['lat'].values,'ro')


F = np.array([[1,0,0,1,0,0],
              [0,1,0,0,1,0],
              [0,0,1,0,0,1],
              [0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

H = np.array([[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0]])
    
R = np.diag([1e-5,1e-5,5])**2

initial_state_mean = np.hstack([measurements[0,:], 3*[0.]])
initial_state_covariance = np.diag([1e-4,1e-4,3.5,1e-4,1e-6,1e-6])**2

kf = KalmanFilter(transition_matrices=F,
                  observation_matrices=H,
                  observation_covariance=R,
                  initial_state_mean=initial_state_mean,
                  initial_state_covariance=initial_state_covariance,
                  em_vars=['transition_covariance'])

kf = kf.em(measurements, n_iter=10)

state_means, state_vars = kf.smooth(measurements)

fig, (ax1, ax2) = plt.subplots(1,2, sharey= True, figsize=(12,7))
ax1.plot(measurements[:,2]), ax2.plot(state_means[:,2])

coords.ix[:,['lon','lat','ele']] = state_means[:,:3]
orig_coords = coords.ix[-coords['idx'].isnull()].set_index('idx')

coords.to_excel('HV.xlsx')

for i, p in enumerate(segment.points):
    p.speed = None
    p.elevation = orig_coords.at[float(i), 'ele']
    p.longitude = orig_coords.at[float(i), 'lon']
    p.latitude = orig_coords.at[float(i), 'lat']

print(segment.get_uphill_downhill())


segment.points[0].speed, segment.points[-1].speed= 0.,0.
gpx_file.add_missing_speeds()
speed = np.array([p.speed for p in segment.points])*3.6
#plt.plot(speed)


    
    
#print(coords.head(2))