#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:53:11 2024

@author: Lucien Bickerstaff @ MPI-BC tSCN
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import math
import scipy
from math import nan
import statistics as stat
import seaborn as sns
import random
from sklearn.linear_model import LinearRegression
from time import sleep
from pyplr import stlab
import string
from pyplr.calibrate import CalibrationContext

# Function to retrieve a small portion of an existing colourmap - to avoid very white and hard to see colours
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

random.seed(4)

plt.rcParams['font.family'] = ['Arial']

####  

'''

fig = plt.figure(figsize=(30, 10))
gs = GridSpec(4, 7, figure = fig, width_ratios=[0.75, 0.75, 0.1, 1, 1, 0.1, 0.9])

#subplots light data
axa = fig.add_subplot(gs[:-2, :-5])
axb = fig.add_subplot(gs[2:, 0])
axc = fig.add_subplot(gs[2:, 1])

#subplots pupil & tickle data
ax1 = fig.add_subplot(gs[:-2, 3])
ax2 = fig.add_subplot(gs[0, 4])
ax3 = fig.add_subplot(gs[1, 4])
ax4 = fig.add_subplot(gs[:-2, 5])
ax5 = fig.add_subplot(gs[:-2, 6])
ax6 = fig.add_subplot(gs[2:, 3])
ax7 = fig.add_subplot(gs[2, 4])
ax8 = fig.add_subplot(gs[3, 4])
ax9 = fig.add_subplot(gs[2:, 5])
ax10 = fig.add_subplot(gs[2:, 6])

ax_separator = fig.add_subplot(gs[:, 2])
ax_separator.axis('off')
ax_separator.axvline(x = 0.5, color = 'black', linestyle = '-', linewidth = 2)

#margins between subplots
plt.subplots_adjust(wspace = 1, hspace=0.7)

'''

fig = plt.figure(figsize=(25, 10))
gs = GridSpec(4, 6, figure = fig, width_ratios=[0.5, 0.5, 1, 1, 0.1, 0.9])

#subplots light data
axa = fig.add_subplot(gs[:-2, :-4])
axb = fig.add_subplot(gs[2:, 0])
axc = fig.add_subplot(gs[2:, 1])

#subplots pupil & tickle data
ax1 = fig.add_subplot(gs[:-2, 2])
ax2 = fig.add_subplot(gs[0, 3])
ax3 = fig.add_subplot(gs[1, 3])
ax4 = fig.add_subplot(gs[:-2, 4])
ax5 = fig.add_subplot(gs[:-2, 5])
ax6 = fig.add_subplot(gs[2:, 2])
ax7 = fig.add_subplot(gs[2, 3])
ax8 = fig.add_subplot(gs[3, 3])
ax9 = fig.add_subplot(gs[2:, 4])
ax10 = fig.add_subplot(gs[2:, 5])

#margins between subplots
plt.subplots_adjust(wspace = 0.7, hspace=0.7)


########################## LIGHT DATA ##########################


chosen_n = 82
logscale = True

"""
Function to:
    - Find timestamp of sneeze events from sneeze log in Notion
    - Plot average illuminance 20 min before and after this timestamp

    - Find timestamp of sneeze events from sneeze log in Notion
    - Use the timestamp as start for light data plotting
    - Calculate mean light levels pre-sneeze (-5 to -2 min) and at sneeze event (-1 to +1 min)
    - Plot categorically, to see step in light intensity before and at sneeze event
"""

random.seed(4)
 
# Opening light data file, getting rid of useless first lines and creating a DataFrame
#with open('D:/photic_sneeze/data/actiwatch_data/actiwatch_combined_data.txt') as old, open('lightdata.txt', 'w') as new:
    #lines = old.readlines()
    #new.writelines(lines[28:])

# Opening combined light data file and creating a DataFrame
data = pd.read_csv('data/actiwatch/actiwatch_combined_data.txt', sep=';')
 
# Stripping DataFrame from unecessary data.
data = pd.DataFrame({'DATE/TIME': data['DATE/TIME'], 'LIGHT': data['LIGHT']})
 
# Sneeze event timestamps from exported Notion sneeze log
date_times = pd.read_csv(r'data\actiwatch\date_times_sneeze_log_notion.csv', sep = ';')
 
event_timestamps = {}
 
for row in date_times.iterrows():
    event_timestamps[row[1][0]]=[]
    # Creates keys with the date values
 
for row in date_times.iterrows():
    event_timestamps[row[1][0]].append(row[1][1])
    # Adds time values to the corresponding date / dict key
    
print(event_timestamps)
 
event_timestamps_list = []
 
for i in range(len(list(event_timestamps.items()))):
    for j in range(len(list(event_timestamps.items())[i][1])):
        event_timestamps_list.append(list(event_timestamps.items())[i][0]+' '+list(event_timestamps.items())[i][1][j])
 
n_sneeze = len(event_timestamps_list)     
 
event_timestamps_min = []
 
for e in event_timestamps_list:
    event_timestamps_min.append(int(e[11:-3])*60 + int(e[14:]))
 
lightdata = pd.DataFrame()
lightdata_calc_sneeze = pd.DataFrame()
 
# Create a new dataframe, add the light values for the desired range for each event
for i in range(n_sneeze):
    event_timestamp = event_timestamps_list[i]
    try:
        index_value = data[data['DATE/TIME'] == event_timestamp+':25']['LIGHT'].index.tolist()[0]
    except IndexError:
        try:
            index_value = data[data['DATE/TIME'] == event_timestamp+':18']['LIGHT'].index.tolist()[0]
        except IndexError:
            index_value = data[data['DATE/TIME'] == event_timestamp+':35']['LIGHT'].index.tolist()[0]
    lightdata[i] = data[index_value-40:index_value+41]['LIGHT'].tolist()
    
lightdata_sneeze_stat = lightdata
 
# Average
lightdata_calc_sneeze['average'] = lightdata.mean(1)
#print(lightdata_calc['average'])
# Std
lightdata_calc_sneeze['std'] = lightdata.std(1)/math.sqrt(len(event_timestamps)) #SEM (= variance...)
#print(lightdata_calc['std'])
 
# Plot average and shaded error bar
axa.plot(np.arange(-20,20.5,0.5), lightdata_calc_sneeze['average'], color='green', label='Sneeze event logged')
axa.fill_between(np.arange(-20,20.5,0.5), 
                 lightdata_calc_sneeze['average']-lightdata_calc_sneeze['std'], 
                 lightdata_calc_sneeze['average']+lightdata_calc_sneeze['std'],
                 facecolor='green',
                 color='green',
                 alpha=0.3)
 
#############################
 
'''
Plot light data when there was no sneeze on top
'''
 
no_sneeze_timestamps = []
 
n_no_sneeze = 1000 # Generating 1000 date & time values to be sure to have enough to work with
 
# while len(no_sneeze_timestamps) < sum([len(event_timestamps[x]) for x in event_timestamps if isinstance(event_timestamps[x], list)]):
while len(no_sneeze_timestamps) < n_no_sneeze:
    
    # Retrieving random date
    num = random.randrange(0, len(event_timestamps), 1)
    date = list(event_timestamps.items())[num][0]
    
    event_timestamps_min = []
    
    # Adding all timestamps within sneeze event for the randomly chosen date in a list of lists
    for e in event_timestamps[date]:
        event_timestamps_min.append(np.arange((int(e[:-3])*60 + int(e[3:]))-20, (int(e[:-3])*60 + int(e[3:]))+20, 1).tolist())
        
    # Merging all timestamps within one single list
    event_timestamps_min_merged = []
        
    # Getting rid of duplicates
    for e in event_timestamps_min:
        event_timestamps_min_merged+=e
        
    event_timestamps_min_merged = list(set(event_timestamps_min_merged))
    
    # Generating random hour and minute timestamp
    num = random.randrange(0, 1440, 1)
    
    rand_timestamps = list(range(num-15, num+15, 1)) # +- 15 min around timestamp
    hour = int(num/60)
    minute = num%60
    
    # Putting minute in right format
    if minute < 10:
        minute = '0'+str(minute)
    else:
        minute = str(minute)
    # Putting hour and date in right format
    if hour < 10:
        hour = '0'+str(hour)
    else:
        hour = str(hour)
    date = str(date)
    
    if int(hour) > 8 and int(hour) < 21: # If random timestamp within wearing time window
    
        if not any(item in rand_timestamps for item in event_timestamps_min_merged): # If random timstamp 30 min time window not in a sneeze event time window
            no_sneeze_timestamps.append(date+' '+hour+':'+minute)
        else:
            print(date+' '+hour+':'+minute+' = Within sneeze event window')
    else:
        print(date+' '+hour+':'+minute+' = Out of recording window')
 
no_sneeze_timestamps = sorted(no_sneeze_timestamps)
#print(no_sneeze_timestamps)
#print(len(no_sneeze_timestamps))
 
lightdata = pd.DataFrame()
lightdata_calc_no_sneeze = pd.DataFrame()
 
# Create a new dataframe, add the light values for the desired range for each event
n = 0
i = 0
while n < chosen_n :
    i+=1
    event_timestamp = no_sneeze_timestamps[i]
    try:
        index_value = data[data['DATE/TIME'] == event_timestamp+':25']['LIGHT'].index.tolist()[0]
        lightdata[str(n)+' - '+event_timestamp] = data[index_value-40:index_value+41]['LIGHT'].tolist()
    except IndexError:
        try:
            index_value = data[data['DATE/TIME'] == event_timestamp+':18']['LIGHT'].index.tolist()[0]
            lightdata[str(n)+' - '+event_timestamp] = data[index_value-40:index_value+41]['LIGHT'].tolist()
            n+=1
        except IndexError:
            try:
                index_value = data[data['DATE/TIME'] == event_timestamp+':35']['LIGHT'].index.tolist()[0]
                lightdata[str(n)+' - '+event_timestamp] = data[index_value-40:index_value+41]['LIGHT'].tolist()
                n+=1
            except IndexError:
                print(str(event_timestamp)+' = Index Error: date chosen out of recording')
              
lightdata_nosneeze_stat = lightdata

#print(lightdata_sneeze_stat)
#print(lightdata_nosneeze_stat)
 
# Average
lightdata_calc_no_sneeze['average'] = lightdata.mean(1)
#print(lightdata_calc['average'])
# Std
lightdata_calc_no_sneeze['std'] = lightdata.std(1)/math.sqrt(len(no_sneeze_timestamps)) #SEM (= variance...)
#print(lightdata_calc['std'])
 
# Plotting

# Figure for plotting
#plt.figure(figsize=(10, 10))
 
# Plot average and shaded error bar
#fig, (axa) = plt.subplots(figsize=(10, 10))
axa.plot(np.arange(-20,20.5,0.5), lightdata_calc_no_sneeze['average'], color='grey', label='No sneeze event logged (reference)')
axa.fill_between(np.arange(-20,20.5,0.5), 
                 lightdata_calc_no_sneeze['average']-lightdata_calc_no_sneeze['std'], 
                 lightdata_calc_no_sneeze['average']+lightdata_calc_no_sneeze['std'],
                 facecolor='grey',
                 color='grey',
                 alpha=0.3)
 
# Plot parameters
if logscale:
    axa.vlines(0, ymin=0, ymax=30000, colors='r', label='Sneeze event')
    axa.set_yscale('log')
    axa.set_ylim(bottom=2e1, top=30000)
    axa.set_xlim(left=-20, right=20)
    axa.set_xlabel('Time (min)', fontsize=10)
    axa.set_ylabel('Illuminance (lx)', fontsize=10)
    # axa.grid(axis = 'both', which = 'major', ls = '--')
    axa.legend(loc='lower left', fontsize=10, handlelength = 1)
    #axa.title('Light data - from 12/07 to 10/08\nAverage light level over 40 minutes +- SEM, N = '+str(n_sneeze))
else:
    axa.vlines(0, ymin=0, ymax=12500, colors='r', label='Sneeze event')
    axa.ylim(bottom=2e1, top=12500)
    axa.xlim(left=-20, right=20)
    axa.xlabel('Time (min)', fontsize=10)
    axa.ylabel('Illuminance (lx)', fontsize=10)
    # axa.grid(axis = 'both', which = 'major', ls = '--')
    axa.legend(loc='lower left', fontsize=10, handlelength = 1)
    #axa.title('Light data - from 12/07 to 10/08\nAverage light level over 40 minutes +- SEM, N = '+str(n_sneeze))
    
##################### STATS

row_sneeze = []
for row in lightdata_sneeze_stat.iterrows():
    row_sneeze.append(list(row))
row_nosneeze = []
for row in lightdata_nosneeze_stat.iterrows():
    row_nosneeze.append(list(row))
    
stats = []
n_tests = 81

for i in range(n_tests):
    stats.append(scipy.stats.ttest_ind(row_sneeze[i][1], row_nosneeze[i][1]))
    
# Bonferroni correction
p_value_threshold = 0.05 / n_tests

for i in range(n_tests):
    if stats[i][1] < p_value_threshold:
        axa.scatter(i/2-20, 12000, color='k')
        
'''
Secondary plots
'''

# Opening combined light data file and creating a DataFrame
data = pd.read_csv('data/actiwatch/actiwatch_combined_data.txt', sep=';')

# Stripping DataFrame from unecessary data.
data = pd.DataFrame({'DATE/TIME': data['DATE/TIME'], 'LIGHT': data['LIGHT']})

# Sneeze event timestamps from exported Notion sneeze log
date_times = pd.read_csv(r'data\actiwatch\date_times_sneeze_log_notion.csv', sep = ';')

event_timestamps = {}

for row in date_times.iterrows():
    event_timestamps[row[1][0]]=[]
    # Creates keys with the date values

for row in date_times.iterrows():
    event_timestamps[row[1][0]].append(row[1][1])
    # Adds time values to the corresponding date / dict key

print(event_timestamps)

event_timestamps_list = []

for i in range(len(list(event_timestamps.items()))):
    for j in range(len(list(event_timestamps.items())[i][1])):
        event_timestamps_list.append(list(event_timestamps.items())[i][0]+' '+list(event_timestamps.items())[i][1][j])

# Pre-sneeze
lightdata_presneeze = pd.DataFrame()
# Create a new dataframe, add the light values for the desired range for each event
for i in range(len(event_timestamps_list)):
    event_timestamp = event_timestamps_list[i]
    try:
        index_value = data[data['DATE/TIME'] == event_timestamp+':25']['LIGHT'].index.tolist()[0]
    except IndexError:
        try:
            index_value = data[data['DATE/TIME'] == event_timestamp+':18']['LIGHT'].index.tolist()[0]
        except IndexError:
            index_value = data[data['DATE/TIME'] == event_timestamp+':35']['LIGHT'].index.tolist()[0]
        
    lightdata_presneeze[i] = data[index_value-10:index_value-4]['LIGHT'].tolist() #-10 = -5 min, -4 = -2 min
# Average
lightdata_presneeze_mean = []
for i in range(len(event_timestamps_list)):
    lightdata_presneeze_mean.append(np.nanmean(lightdata_presneeze[i]))

# At sneeze
lightdata_atsneeze = pd.DataFrame()
# Create a new dataframe, add the light values for the desired range for each event
for i in range(len(event_timestamps_list)):
    event_timestamp = event_timestamps_list[i]
    try:
        index_value = data[data['DATE/TIME'] == event_timestamp+':25']['LIGHT'].index.tolist()[0]
    except IndexError:
        try:
            index_value = data[data['DATE/TIME'] == event_timestamp+':18']['LIGHT'].index.tolist()[0]
        except IndexError:
            index_value = data[data['DATE/TIME'] == event_timestamp+':35']['LIGHT'].index.tolist()[0]
    lightdata_atsneeze[i] = data[index_value-2:index_value+2]['LIGHT'].tolist() #-2 = -1 min, +2 = +1 min
# Average
lightdata_atsneeze_mean = []
for i in range(len(event_timestamps_list)):
    lightdata_atsneeze_mean.append(np.nanmean(lightdata_atsneeze[i]))

x = ['Pre-sneeze\n-5 to -2 min', 'At sneeze\n-1 to +1 min']

# Plot mean and shaded error bar
axb.plot(x, [lightdata_presneeze_mean, lightdata_atsneeze_mean], color='green', alpha=0.5)
axb.plot(x, [lightdata_presneeze_mean, lightdata_atsneeze_mean], 'ko')
axb.set_yticks([1e1, 1e2, 1e3, 1e4], labels=['10', '100', '1000', '10000'])
axb.set_yscale('log')
axb.set_ylabel('Illuminance (lx)', fontsize=10)
# axb.grid(axis = 'y', which = 'major', ls = '--')
#axb.set_title('Individual raw mean values')

# Contrast
lightdata_presneeze_mean_percent = []
lightdata_atsneeze_mean_percent = []

for i in range(len(event_timestamps_list)):
    lightdata_presneeze_mean_percent.append(lightdata_presneeze_mean[i]/lightdata_presneeze_mean[i]*100)
    lightdata_atsneeze_mean_percent.append(lightdata_atsneeze_mean[i]/lightdata_presneeze_mean[i]*100)

# Recreating DataFrame from the tzo lists from plotting afterwards
zipped = list(zip(lightdata_presneeze_mean_percent, lightdata_atsneeze_mean_percent))
df = pd.DataFrame(zipped, columns = x)

# Dropping 13th value = 86 000 %
df = df.drop(index=13)

# Dropping Pre-sneeze data (might be useful to keep the somewhere if needed)
df2 = df.drop(['Pre-sneeze\n-5 to -2 min'], axis=1)

#axc = pt.RainCloud(data = df2/100) #/100 to have contrast in %
sns.stripplot(data = df2/100, color='black', ax=axc)
sns.boxplot(data = df2/100, width=0.5, palette='Greens', ax=axc)
axc.set_ylabel('x Pre-sneeze\n-5 to -2 min', fontsize=10)
#axc.set_title('Contrast')
axc.set_yscale('log')
axc.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3], labels=['0.1x', '1x', '10x', '100x', '1000x'])
# axc.grid(axis = 'y', which = 'major', ls = '--')

#fig.suptitle('Light data - from 12/07 to 10/08\nPre-sneeze and at sneeze event, N = '
             #+str(len(event_timestamps_list)))

# Stats
stats = scipy.stats.ttest_rel(lightdata_presneeze_mean, lightdata_atsneeze_mean)
p_value = round(stats[1], 13)
axb.text(.5, 25, 'P = '+str(p_value), fontsize=10)

#titles
axa.set_title('Light exposure data', loc='left', fontweight='bold')
axb.set_title('Contrast', loc='left', fontweight='bold')
axc.set_title('Contrast', loc='left', fontweight='bold')

########################## SPECTRUM ########################## 

cmap = plt.get_cmap('YlOrBr')
cmap = truncate_colormap(cmap, 0.5, 1)

cc = CalibrationContext('data/calibration/S2_corrected_oo_spectra.csv', binwidth = 1)

intensity_settings = [65, 130, 650, 2405]
lx_values = [440, 1100, 4400, 17600]
for i in range(4):
    spd = cc.predict_spd(intensities = [intensity_settings[i]]*10)
    ax1.plot(spd.values.flatten().tolist(), color = cmap(i/4), label = str(lx_values[i])+' lx')

ax1.set_ylabel('SPD (W/m2/nm)', fontsize=10)
ax1.set_xlabel('Wavelength (nm)', fontsize=10)
ax1.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], ['380', '430', '480', '530', '580', '630', '680', '730', '780'])
ax1.set_title('Spectral distribution', loc='left', fontweight='bold')
# ax1.grid(axis = 'both', which = 'major', ls = '--')

ax1.legend(loc='upper right', fontsize = 10, handlelength = 1)._legend_box.align = 'left'

########################## PUPIL RAW ########################## 

### one-shot

conf = 0.7
moving_average_window = 40

# Specifying path
path = 'data/one_shot_exp/'
dirs = os.listdir(path)#[:-1] # minus test directory
dirs.pop(25) # minus trial 25 which doesn't have sneeze data for some reason
dirs.pop(52) # minus trial 53 which doesn't have sneeze data for some reason

data_60 = pd.DataFrame()
data_150 = pd.DataFrame()
data_600 = pd.DataFrame()
data_2400 = pd.DataFrame()

num = 0

for i in dirs:
    num += 1
    # Opening csv files
    data = pd.read_csv(path+str(i)+'/exports/000/pupil_positions.csv')
    annotations = pd.read_csv(path+str(i)+'/exports/000/annotations.csv')
    info = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
    
    # Apply confidence threshold on pupil diameter values
    data['diameter_3d'].mask(data['confidence']<conf, other = nan, inplace = True)
               
    # Stripping DataFrame from unecessary data. Converting timestamps in min and normalising to light onset = 0 min
    data = pd.DataFrame({'pupil_timestamp': data['pupil_timestamp']/60-annotations.timestamp[annotations['label']=='lights on'].to_list()[0]/60,
                         'diameter_3d': data['diameter_3d']})
    #print(data)
    
    # GETTING RID OF DIAMETER VALUES WHERE TIMESTAMP IS UNDER -1 OR OVER +1.5:
    data = data.mask((-1>data['pupil_timestamp']) | (data['pupil_timestamp']>1.5), other = 'trash') # labelling unwanted values as 'trash'
    data.drop(index=data[data['pupil_timestamp'] == 'trash'].index, inplace = True) # getting rid of them
    data = data.reset_index(drop=True) # resetting index to 0
    
    if (info['setting'] == 60)[0]:
        data_60 = pd.concat([data_60, data['diameter_3d']], axis = 1)
    elif (info['setting'] == 150)[0]:
        data_150 = pd.concat([data_150, data['diameter_3d']], axis = 1)
    elif (info['setting'] == 600)[0]:
        data_600 = pd.concat([data_600, data['diameter_3d']], axis = 1)
    elif (info['setting'] == 2400)[0]:
        data_2400 = pd.concat([data_2400, data['diameter_3d']], axis = 1)
    else:
        print('error')
        
    print(str(num)+'/'+str(len(dirs)))
    
    # Time of light onset, in seconds
    #onset_timestamp = (annotations.timestamp[0]/60)-min(data['pupil_timestamp'])
    #print(onset_timestamp)
    #print(data['pupil_timestamp'])

# Filling DataFrame
average_data = pd.DataFrame()
average_data['60 mean'] = data_60.mean(1)
average_data['60 std'] = data_60.std(1)
average_data['150 mean'] = data_150.mean(1)
average_data['150 std'] = data_150.std(1)
average_data['600 mean'] = data_600.mean(1)
average_data['600 std'] = data_600.std(1)
average_data['2400 mean'] = data_2400.mean(1)
average_data['2400 std'] = data_2400.std(1)

cmap = plt.get_cmap('Reds')

# Moving average
average_data['60 mean'] = average_data['60 mean'].rolling(window=moving_average_window, min_periods=2).mean()
average_data['150 mean'] = average_data['150 mean'].rolling(window=moving_average_window, min_periods=2).mean()
average_data['600 mean'] = average_data['600 mean'].rolling(window=moving_average_window, min_periods=2).mean()
average_data['2400 mean'] = average_data['2400 mean'].rolling(window=moving_average_window, min_periods=2).mean()

# Plotting

ax2.plot(average_data['60 mean'], color = cmap(0.4))
ax2.plot(average_data['150 mean'], color = cmap(0.6))
ax2.plot(average_data['600 mean'], color = cmap(0.8))
ax2.plot(average_data['2400 mean'], color = cmap(0.99))

# plot parameters
ax2.set_xticks(ticks = [0, 
round(len(average_data.index)/5), 
round(len(average_data.index)/5)*2,
round(len(average_data.index)/5)*3,
round(len(average_data.index)/5)*4,
len(average_data.index)], labels = ['-60', '-30', '0', '30', '60', '90'])
ax2.set_ylabel('Pupil diameter (mm)', fontsize=10)
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_xlim(left=0, right=len(average_data.index))
ax2.set_ylim(bottom=0, top=6)
ax2.vlines(round(len(average_data.index)/5)*2, ymin=0, ymax=10, colors='red', label='Light on')
# ax2.grid(axis = 'both', which = 'major', ls = '--')

ax2.legend(loc='lower left', fontsize = 10, handlelength = 1)


#### 30-min exp

# Specifying path
path = 'data/30_min_exp/'
dirs = os.listdir(path)#[:-2] # minus test sessions

data_0 = pd.DataFrame()
data_60 = pd.DataFrame()
data_150 = pd.DataFrame()
data_600 = pd.DataFrame()
data_2400 = pd.DataFrame()

num = 0

for i in dirs:
    
    num += 1
    n_trials = 24
    
    print('######### '+str(num)+'/'+str(len(dirs))+' #########')
    
    # Opening csv files
    data = pd.read_csv(path+str(i)+'/exports/000/pupil_positions.csv')
    annotations = pd.read_csv(path+str(i)+'/exports/000/annotations.csv')
    info = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
    
    annotations_timestamps = annotations.timestamp[annotations['label']=='lights on'].to_list()
    
    # Apply confidence threshold on pupil diameter values
    data['diameter_3d'].mask(data['confidence']<conf, other = nan, inplace = True)
    
    setting_list = info['setting'][0][1:-1].split(', ')
    
    for j in range(n_trials):
        
        annotation_timestamp = annotations_timestamps[j]
        #print(setting_list[j])
                        
        # Stripping DataFrame from unecessary data. Converting timestamps in min and normalising to light onset = 0 min
        data_temp = pd.DataFrame({'pupil_timestamp': data['pupil_timestamp']/60-annotation_timestamp/60,
                             'diameter_3d': data['diameter_3d']})
        #print(data)
        
        # GETTING RID OF DIAMETER VALUES WHERE TIMESTAMP IS UNDER -1 OR OVER +1.5:
        data_temp = data_temp.mask((0.05>data_temp['pupil_timestamp']) | (data_temp['pupil_timestamp']>0.55), other = 'trash') # labelling unwanted values as 'trash'
        data_temp.drop(index=data_temp[data_temp['pupil_timestamp'] == 'trash'].index, inplace = True) # getting rid of them
        data_temp = data_temp.reset_index(drop=True) # resetting index to 0
        
        if (setting_list[j] == '-2'):
            data_0 = pd.concat([data_0, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '-1'):
            data_60 = pd.concat([data_60, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '0'):
            data_150 = pd.concat([data_150, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '1'):
            data_600 = pd.concat([data_600, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '2'):
            data_2400 = pd.concat([data_2400, data_temp['diameter_3d']], axis = 1)
        else:
            print('error')
        
        print(str(j+1)+'/'+str(n_trials))                      
              
# Filling DataFrame
average_data = pd.DataFrame()
average_data['0 mean'] = data_0.mean(1)
average_data['0 std'] = data_0.std(1)
average_data['60 mean'] = data_60.mean(1)
average_data['60 std'] = data_60.std(1)
average_data['150 mean'] = data_150.mean(1)
average_data['150 std'] = data_150.std(1)
average_data['600 mean'] = data_600.mean(1)
average_data['600 std'] = data_600.std(1)
average_data['2400 mean'] = data_2400.mean(1)
average_data['2400 std'] = data_2400.std(1)

cmap = plt.get_cmap('Blues')

# Moving average
average_data['0 mean'] = average_data['0 mean'].rolling(window=moving_average_window, min_periods=2).mean()
average_data['60 mean'] = average_data['60 mean'].rolling(window=moving_average_window, min_periods=2).mean()
average_data['150 mean'] = average_data['150 mean'].rolling(window=moving_average_window, min_periods=2).mean()
average_data['600 mean'] = average_data['600 mean'].rolling(window=moving_average_window, min_periods=2).mean()
average_data['2400 mean'] = average_data['2400 mean'].rolling(window=moving_average_window, min_periods=2).mean()

# Plotting

ax3.plot(average_data['0 mean'], color = cmap(0.4))
ax3.plot(average_data['60 mean'], color = cmap(0.55))
ax3.plot(average_data['150 mean'], color = cmap(0.7))
ax3.plot(average_data['600 mean'], color = cmap(0.85))
ax3.plot(average_data['2400 mean'], color = cmap(0.99))



# plot parameters
ax3.set_xticks(ticks = [0, 
round(len(average_data.index)/6), 
round(len(average_data.index)/6)*2,
round(len(average_data.index)/6)*3,
round(len(average_data.index)/6)*4,
round(len(average_data.index)/6)*5,
len(average_data.index)], labels = ['0', '5', '10', '15', '20', '25', '30'])
ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('Pupil diameter (mm)', fontsize=10)
ax3.set_xlim(left=0, right=len(average_data.index))
ax3.set_ylim(bottom=0, top=6)
# ax3.grid(axis = 'both', which = 'major', ls = '--')

## arrows and text boxes

arrow_props1 = dict(facecolor = 'black', edgecolor = 'black', arrowstyle = '->')
ax2.annotate('', xy = (33500, 2.3), xytext = (0.89, 0.1), transform = ax2.transAxes, textcoords = 'axes fraction', arrowprops = arrow_props1)

text_box1 = dict(boxstyle = 'round', facecolor='white', alpha = 1, edgecolor = 'none')
ax2.text(0.89, 0.1, '17600 lx', transform = ax2.transAxes, fontsize = 10, bbox = text_box1, ha = 'center')

arrow_props2 = dict(facecolor = 'black', edgecolor = 'black', arrowstyle = '->')
ax2.annotate('', xy = (31000, 4.2), xytext = (0.75, 0.85), transform = ax2.transAxes, textcoords = 'axes fraction', arrowprops = arrow_props2)

text_box2 = dict(boxstyle = 'round', facecolor='white', alpha = 1, edgecolor = 'none')
ax2.text(0.75, 0.85, '440 lx', transform = ax2.transAxes, fontsize = 10, bbox = text_box2, ha = 'center')

##

arrow_props3 = dict(facecolor = 'black', edgecolor = 'black', arrowstyle = '->')
ax3.annotate('', xy = (4400, 0.7), xytext = (0.7, 0.8), transform = ax3.transAxes, textcoords = 'axes fraction', arrowprops = arrow_props3)

text_box3 = dict(boxstyle = 'round', facecolor='white', alpha = 1, edgecolor = 'none')
ax3.text(0.7, 0.8, '17600 lx', transform = ax3.transAxes, fontsize = 10, bbox = text_box3, ha = 'center')

arrow_props4 = dict(facecolor = 'black', edgecolor = 'black', arrowstyle = '->')
ax3.annotate('', xy = (2500, 3.1), xytext = (0.3, 0.8), transform = ax3.transAxes, textcoords = 'axes fraction', arrowprops = arrow_props4)

text_box4 = dict(boxstyle = 'round', facecolor='white', alpha = 1, edgecolor = 'none')
ax3.text(0.3, 0.8, '0 lx', transform = ax3.transAxes, fontsize = 10, bbox = text_box4, ha = 'center')

##

# final touches
ax2.set_title('Pupil data - One-shot experiment', loc='left', fontweight='bold')
ax3.set_title('Pupil data - 30-min experiment', loc='left', fontweight='bold')




########################## PUPIL AVERAGE ##########################


#### one shot

random.seed(4)

# Specifying path
path = 'data/one_shot_exp/'
dirs = os.listdir(path)#[:-1] # minus test directory
dirs.pop(25) # minus trial 25 which doesn't have sneeze data for some reason
dirs.pop(52) # minus trial 53 which doesn't have sneeze data for some reason

data_0 = pd.DataFrame()
data_60 = pd.DataFrame()
data_150 = pd.DataFrame()
data_600 = pd.DataFrame()
data_2400 = pd.DataFrame()

num = 0

for i in dirs:
    num += 1
    # Opening csv files
    data = pd.read_csv(path+str(i)+'/exports/000/pupil_positions.csv')
    annotations = pd.read_csv(path+str(i)+'/exports/000/annotations.csv')
    info = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
    
    # Apply confidence threshold on pupil diameter values
    data['diameter_3d'].mask(data['confidence']<conf, other = nan, inplace = True)
               
    # Stripping DataFrame from unecessary data. Converting timestamps in min and normalising to light onset = 0 min
    data = pd.DataFrame({'pupil_timestamp': data['pupil_timestamp']/60-annotations['timestamp'][0]/60, 'diameter_3d': data['diameter_3d']})
    #print(data)
    
    # GETTING RID OF DIAMETER VALUES WHERE TIMESTAMP IS UNDER -1 OR OVER 0 (= keep last minute of dark period):
    data_off = data.mask((-1>data['pupil_timestamp']) | (data['pupil_timestamp']>0), other = 'trash') # labelling unwanted values as 'trash'
    data_off.drop(index=data_off[data_off['pupil_timestamp'] == 'trash'].index, inplace = True) # getting rid of them
    data_off = data_off.reset_index(drop=True) # resetting index to 0
    
    data_0 = pd.concat([data_0, data_off['diameter_3d']], axis = 1)
    
    # GETTING RID OF DIAMETER VALUES WHERE TIMESTAMP IS UNDER 0.25 OR OVER 0.75 (= keep plateau):
    data_on = data.mask((0.25>data['pupil_timestamp']) | (data['pupil_timestamp']>0.75), other = 'trash') # labelling unwanted values as 'trash'
    data_on.drop(index=data_on[data_on['pupil_timestamp'] == 'trash'].index, inplace = True) # getting rid of them
    data_on = data_on.reset_index(drop=True) # resetting index to 0
    
    if (info['setting'] == 60)[0]:
        data_60 = pd.concat([data_60, data_on['diameter_3d']], axis = 1)
    elif (info['setting'] == 150)[0]:
        data_150 = pd.concat([data_150, data_on['diameter_3d']], axis = 1)
    elif (info['setting'] == 600)[0]:
        data_600 = pd.concat([data_600, data_on['diameter_3d']], axis = 1)
    elif (info['setting'] == 2400)[0]:
        data_2400 = pd.concat([data_2400, data_on['diameter_3d']], axis = 1)
    else:
        print('error')
    
    print(str(num)+'/'+str(len(dirs)))
    # Time of light onset, in seconds
    #onset_timestamp = (annotations.timestamp[0]/60)-min(data['pupil_timestamp'])
    #print(onset_timestamp)
    #print(data['pupil_timestamp'])

# Average
average = [data_0.mean().mean(), 
           data_60.mean().mean(), 
           data_150.mean().mean(),
           data_600.mean().mean(),
           data_2400.mean().mean()]
print('Average: '+str(average))
# Std
std = [data_0.stack().std(),
      data_60.stack().std(),
      data_150.stack().std(),
      data_600.stack().std(),
      data_2400.stack().std()]
print('STD: '+str(std))

intensities = [1.00000000e+00, 440, 1099, 4396, 17582]
intensities_reg = intensities[1:]
average_reg = average[1:]

log_intensities = []
for i in intensities_reg:
    log_intensities.append(math.log(i, 10))
xrange = [300, 30000]
log_xrange=[]
for i in xrange:
    log_xrange.append(math.log(i, 10))

# Linear regression
x = np.array(log_intensities).reshape((-1, 1))
y = np.array(average_reg)
model = LinearRegression()
model.fit(x, y)
print(f"slope: {model.coef_}")
y_pred = model.predict(np.array(log_xrange).reshape((-1, 1)))
#print(y_pred)

n_labels = [str(len(data_0.columns)), str(len(data_60.columns)), str(len(data_150.columns)), str(len(data_600.columns)), str(len(data_2400.columns))]

# Plotting

ax4.spines.right.set_visible(False)
ax4.scatter((intensities[0]*1.005), average[0], c='r', marker='v', alpha=0.7)
ax4.errorbar((intensities[0]*1.005), average[0], std[0], ecolor='r', capsize=4, fmt='none', alpha=0.7)
ax4.set_xlim(0.95, 1.05)
ax4.set_ylim(bottom = 0, top = 6)
ax4.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax4.set_xticks([1], ['0'])
# ax4.grid(axis = 'both', which = 'both', ls = '--')
ax4.set_ylabel('Pupil diameter (mm)', fontsize=10)

ax5.set_xlim(300, 30000)
ax5.spines.left.set_visible(False)
ax5.scatter([x*1.03 for x in intensities_reg], average_reg, c='r', marker='v', alpha=0.7, label='One-shot')
ax5.errorbar([x*1.03 for x in intensities], average, std, ecolor='r', capsize=4, fmt='none', alpha=0.7)
ax5.plot(xrange, y_pred, color='r', linestyle='dashed', alpha=0.7)
ax5.set_xscale('log')
ax5.set_ylim(bottom = 0, top = 6)
ax5.set_xticks([300, 1000, 3000, 10000, 30000], labels = [300, 1000, 3000, 10000, 30000])
# ax5.grid(axis = 'both', which = 'both', ls = '--')
ax5.tick_params(labelleft=False, left=False)
ax5.set_xlabel('Illuminance (lx)', fontsize=10)

### 30 min

# Specifying path
path = 'data/30_min_exp/'
dirs = os.listdir(path)#[:-2] # minus test sessions

data_0 = pd.DataFrame()
data_60 = pd.DataFrame()
data_150 = pd.DataFrame()
data_600 = pd.DataFrame()
data_2400 = pd.DataFrame()

num = 0

for i in dirs:
    
    num += 1
    n_trials = 24
    
    print('######### '+str(num)+'/'+str(len(dirs))+' #########')
    
    # Opening csv files
    data = pd.read_csv(path+str(i)+'/exports/000/pupil_positions.csv')
    annotations = pd.read_csv(path+str(i)+'/exports/000/annotations.csv')
    info = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
    
    annotations_timestamps = annotations.timestamp[annotations['label']=='lights on'].to_list()
    
    # Apply confidence threshold on pupil diameter values
    data['diameter_3d'].mask(data['confidence']<conf, other = nan, inplace = True)
    
    setting_list = info['setting'][0][1:-1].split(', ')
    
    for j in range(n_trials):
        
        annotation_timestamp = annotations_timestamps[j]
        #print(setting_list[j])
                        
        # Stripping DataFrame from unecessary data. Converting timestamps in min and normalising to light onset = 0 min
        data_temp = pd.DataFrame({'pupil_timestamp': data['pupil_timestamp']/60-annotation_timestamp/60,
                             'diameter_3d': data['diameter_3d']})
        #print(data)
        
        # GETTING RID OF DIAMETER VALUES WHERE TIMESTAMP IS UNDER -1 OR OVER +1.5:
        data_temp = data_temp.mask((0.05>data_temp['pupil_timestamp']) | (data_temp['pupil_timestamp']>0.55), other = 'trash') # labelling unwanted values as 'trash'
        data_temp.drop(index=data_temp[data_temp['pupil_timestamp'] == 'trash'].index, inplace = True) # getting rid of them
        data_temp = data_temp.reset_index(drop=True) # resetting index to 0
        
        if (setting_list[j] == '-2'):
            data_0 = pd.concat([data_0, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '-1'):
            data_60 = pd.concat([data_60, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '0'):
            data_150 = pd.concat([data_150, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '1'):
            data_600 = pd.concat([data_600, data_temp['diameter_3d']], axis = 1)
        elif (setting_list[j] == '2'):
            data_2400 = pd.concat([data_2400, data_temp['diameter_3d']], axis = 1)
        else:
            print('error')
        
        print(str(j+1)+'/'+str(n_trials))

# Average
average = [data_0.mean().mean(), 
            data_60.mean().mean(), 
            data_150.mean().mean(),
            data_600.mean().mean(),
            data_2400.mean().mean(),]
print('Average: '+str(average))
# Std
std = [data_0.stack().std(),
      data_60.stack().std(),
      data_150.stack().std(),
      data_600.stack().std(),
      data_2400.stack().std()]
print('STD: '+str(std))

n_labels = [str(len(data_0.columns)), str(len(data_60.columns)), str(len(data_150.columns)), str(len(data_600.columns)), str(len(data_2400.columns))]

intensities = [1.00000000e+00, 440, 1099, 4396, 17582]
intensities_reg = intensities[1:]
average_reg = average[1:]

log_intensities = []
for i in intensities_reg:
    log_intensities.append(math.log(i, 10))
xrange = [300, 30000]
log_xrange=[]
for i in xrange:
    log_xrange.append(math.log(i, 10))

# Linear regression
x = np.array(log_intensities).reshape((-1, 1))
y = np.array(average_reg)
model = LinearRegression()
model.fit(x, y)
print(f"slope: {model.coef_}")
y_pred = model.predict(np.array(log_xrange).reshape((-1, 1)))
    
# Plotting 
# Plot parameters

ax4.scatter((intensities[0]*0.995), average[0], c='b', alpha=0.7)
ax4.errorbar((intensities[0]*0.995), average[0], std[0], ecolor='b', capsize=4, fmt='none', alpha=0.7)

ax5.scatter([x*0.97 for x in intensities_reg], average_reg, c='b', alpha=0.7, label='30-min')
ax5.errorbar([x*0.97 for x in intensities], average, std, ecolor='b', capsize=4, fmt='none', alpha=0.7)
ax5.plot(xrange, y_pred, color='b', linestyle='dashed', alpha=0.7)
ax5.legend(loc='upper right', fontsize = 10)._legend_box.align = 'left'

ax4.set_title('Pupil data - Mean +-STD', loc='left', fontweight='bold')


########################## DIAGRAM ##########################


# ax6.grid(axis = 'both', which = 'both', ls = '--')



########################## TICKLE RAW ########################## 

# Red =  one-shot exp, Blue = 30-min exp

# one shot exp

# Specifying path
path = 'data/one_shot_exp/'
dirs = os.listdir(path)#[:-1] # minus test directory
dirs.pop(25) # trial 025 failed. Pupil recording worked but not tickle rating...
dirs.pop(52) # trial 053 failed. Pupil recording worked but not tickle rating...

list60 = []
list150 = []
list600 = []
list2400 = []

for i in dirs:

    # Opening csv file
    data = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')

    if (data['setting'] == 60)[0]:
        #print(data['reponse_q2'][0])
        list60.append(int(data['reponse_q2'][0]))
    elif (data['setting'] == 150)[0]:
        #print(data['reponse_q2'][0])
        list150.append(int(data['reponse_q2'][0]))
    elif (data['setting'] == 600)[0]:
        #print(data['reponse_q2'][0])
        list600.append(int(data['reponse_q2'][0]))
    elif (data['setting'] == 2400)[0]:
        #print(data['reponse_q2'][0])
        list2400.append(int(data['reponse_q2'][0]))
    else:
        print('error')

intensities = ['440', '1099', '4396', '17582']

listall = [list60, list150, list600, list2400]

for i in listall: # sorting lists
    i.sort()

cmap = plt.get_cmap('Reds')
cmap = truncate_colormap(cmap, 0.25, 1)

# Plot parameters
for i in range(len(listall)): # intensity
    n=0
    for j in range(len(listall[i])):
        ax7.barh(intensities[i],
            1/len(listall[i]),
            left=n, 
            color=cmap(listall[i][j]/10))
        n+=1/len(listall[i])

handles = [None]*11
colours =  [None]*11

for i in range(11): # Generating blank values for each level, with the corresponding label and color for the legend
    colours[i] = cmap(i/10)
    ax7.barh(intensities[0],
             0,
             color=colours[i])  # ,label=str(i)+'/10'
    if i == 0 or i ==10:
        handles[i] = str(i)
    else:
        handles[i] = ''

legend_handles = [Line2D([0], [0], color = color, label = label, linewidth=6) for color, label in zip(colours, handles)]

# Plot parameters
ax7.set_ylabel('Illuminance (lx)', fontsize=10)
ax7.set_xlabel('Proportion', fontsize=10)
ax7.set_yticks(intensities, labels=['440', '1100', '4400', '17600'])
ax7.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
ax7.set_title('Tickle data - One-shot experiment', loc='left', fontweight='bold')
ax7.legend(handles = legend_handles, loc=[1.03, -0.2], fontsize = 10, labelspacing = 0.1, handlelength = 1)._legend_box.align = 'left'

# 30 min exp

# Specifying path
path = 'data/30_min_exp/'
dirs = os.listdir(path)#[:-2] # minus test sessions

list0 = []
list60 = []
list150 = []
list600 = []
list2400 = []

n_trials = 24

for i in dirs:
    
    # Opening csv file
    data = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
    
    setting_list = data['setting'][0][1:-1].split(', ')
    
    for j in range(n_trials):
    
        if (setting_list[j] == '-2'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list0.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '-1'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list60.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '0'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list150.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '1'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list600.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '2'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list2400.append(int(data['reponse_q'+str(3*j+2)][0]))
        else:
            print('error')

intensities = ['0', '440', '1099', '4396', '17582']

listall = [list0, list60, list150, list600, list2400]

for i in listall: # sorting lists
    i.sort()
    
cmap = plt.get_cmap('Blues')
cmap = truncate_colormap(cmap, 0.25, 1)

# Plot parameters
for i in range(len(listall)): # intensity
    n=0
    for j in range(len(listall[i])):
        ax8.barh(intensities[i], 
                1/len(listall[i]), 
                left=n, 
                color=cmap(listall[i][j]/10))
        n+=1/len(listall[i])

handles = [None]*11
colours =  [None]*11

for i in range(11): # Generating blank values for each level, with the corresponding label and color for the legend
    colours[i] = cmap(i/10)
    ax8.barh(intensities[0],
             0,
             color=colours[i])  # ,label=str(i)+'/10'
    if i == 0 or i ==10:
        handles[i] = str(i)
    else:
        handles[i] = ''

legend_handles = [Line2D([0], [0], color = color, label = label, linewidth=6) for color, label in zip(colours, handles)]


# Plot parameters
ax8.set_ylabel('Illuminance (lx)', fontsize=10)
ax8.set_xlabel('Proportion', fontsize=10)
ax8.set_yticks(intensities, labels=['0', '440', '1100', '4400', '17600'])
ax8.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
ax8.set_title('Tickle data - 30-min experiment', loc='left', fontweight='bold')
ax8.legend(handles = legend_handles, loc=[1.03, -0.2], fontsize = 10, labelspacing = 0.1, handlelength = 1)._legend_box.align = 'left'



########################## TICKLE SUM ########################## 



#### one shot

random.seed(4)

# Specifying path
path = 'data/one_shot_exp/'
dirs = os.listdir(path)#[:-1] # minus test directory
dirs.pop(25) # trial 025 failed. Pupil recording worked but not tickle rating...
dirs.pop(52) # trial 053 failed. Pupil recording worked but not tickle rating...

list60 = []
list150 = []
list600 = []
list2400 = []

for i in dirs:
    # Opening csv file
    data = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
    
    if (data['setting'] == 60)[0]:
        #print(data['reponse_q2'][0])
        list60.append(int(data['reponse_q2'][0]))
    elif (data['setting'] == 150)[0]:
        #print(data['reponse_q2'][0])
        list150.append(int(data['reponse_q2'][0]))
    elif (data['setting'] == 600)[0]:
        #print(data['reponse_q2'][0])
        list600.append(int(data['reponse_q2'][0]))
    elif (data['setting'] == 2400)[0]:
        #print(data['reponse_q2'][0])
        list2400.append(int(data['reponse_q2'][0]))
    else:
        print('error')
    
# Average
average = [stat.mean(list60), stat.mean(list150), stat.mean(list600), stat.mean(list2400)]
# Std
std = [stat.stdev(list60), stat.stdev(list150), stat.stdev(list600), stat.stdev(list2400)]

intensities = [440, 1099, 4396, 17582]
log_intensities = []
for i in intensities:
    log_intensities.append(math.log(i, 10))
xrange = [300, 30000]
log_xrange=[]
for i in xrange:
    log_xrange.append(math.log(i, 10))

# Linear regression
x = np.array(log_intensities).reshape((-1, 1))
y = np.array(average)
model = LinearRegression()
model.fit(x ,y)
print(f"slope: {model.coef_}")
y_pred = model.predict(np.array(log_xrange).reshape((-1, 1)))
#print(y_pred)

n_labels = [str(len(list60)), str(len(list150)), str(len(list600)), str(len(list2400))]
    
# Plotting
ax10.scatter([x*1.03 for x in intensities], average, c = 'r', marker='v', alpha=0.7, label='One-shot')
ax10.plot(xrange, y_pred, color='r', linestyle='dashed', alpha=0.7)

# Plot parameters
ax10.errorbar([x*1.03 for x in intensities], average, std, ecolor='r', capsize=4, fmt='none', alpha=0.7)
ax10.set_xscale('log')
ax10.set_ylim(bottom=0, top=10)
ax10.set_xlim(left=300, right=30000)
ax10.set_xlabel('Illuminance (lx)', fontsize=10)
ax10.set_xticks([300, 1000, 3000, 10000, 30000], labels=[300, 1000, 3000, 10000, 30000])
ax10.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



#### 30 min exp



# Specifying path
path = 'data/30_min_exp/'
dirs = os.listdir(path)#[:-2] # minus test sessions

list0 = []
list60 = []
list150 = []
list600 = []
list2400 = []

n_trials = 24

for i in dirs:
    
    # Opening csv file
    data = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
    
    setting_list = data['setting'][0][1:-1].split(', ')
    
    for j in range(n_trials):
    
        if (setting_list[j] == '-2'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list0.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '-1'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list60.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '0'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list150.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '1'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list600.append(int(data['reponse_q'+str(3*j+2)][0]))
        elif (setting_list[j] == '2'):
            #print(data['reponse_q'+str(3*j+2)][0])
            list2400.append(int(data['reponse_q'+str(3*j+2)][0]))
        else:
            print('error')
    
# Average
average = [stat.mean(list0), stat.mean(list60), stat.mean(list150), stat.mean(list600), stat.mean(list2400)]
# Std
std = [stat.stdev(list0), stat.stdev(list60), stat.stdev(list150), stat.stdev(list600), stat.stdev(list2400)]

intensities = [0, 440, 1099, 4396, 17582]
intensities_reg = intensities[1:]
average_reg = average[1:]

log_intensities = []
for i in intensities_reg:
    log_intensities.append(math.log(i, 10))
xrange = [300, 30000]
log_xrange=[]
for i in xrange:
    log_xrange.append(math.log(i, 10))

# Linear regression
x = np.array(log_intensities).reshape((-1, 1))
y = np.array(average_reg)
model = LinearRegression()
model.fit(x ,y)
print(f"slope: {model.coef_}")
y_pred = model.predict(np.array(log_xrange).reshape((-1, 1)))

n_labels = [str(len(list0)), str(len(list60)), str(len(list150)), str(len(list600)), str(len(list2400))]   

# Plotting 

ax9.spines.right.set_visible(False)
ax9.scatter(intensities[0], average[0], c='b', alpha=0.7)
ax9.errorbar(intensities[0], average[0], std[0], ecolor='b', capsize=4, fmt='none', alpha=0.7)
ax9.set_ylim(bottom = 0, top = 10)
ax9.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# ax9.grid(axis = 'both', which = 'both', ls = '--')
ax9.set_ylabel('Tickle rating /10', fontsize=10)
ax9.set_xticks([0], ['0'])
ax10.spines.left.set_visible(False)
ax10.tick_params(labelleft=False, left=False)
ax10.scatter([x*0.97 for x in intensities_reg], average_reg, c='b', alpha=0.7, label='30-min')
ax10.errorbar([x*0.97 for x in intensities], average, std, ecolor='b', capsize=4, fmt='none', alpha=0.7)
ax10.plot(xrange, y_pred, color='b', linestyle='dashed', alpha=0.7)
# ax10.grid(axis = 'both', which = 'both', ls = '--')
ax10.legend(loc='lower right', fontsize = 10)

ax9.set_title('Tickle sensation ratings - Mean +-STD', loc='left', fontweight='bold')


####

axa.annotate('A', xy=(-0.113, 1.1), xycoords='axes fraction', size=20, weight='bold', va='top')
axb.annotate('B', xy=(-0.333, 1.1), xycoords='axes fraction', size=20, weight='bold', va='top')
axc.annotate('C', xy=(-0.333, 1.1), xycoords='axes fraction', size=20, weight='bold', va='top')
ax1.annotate('D', xy=(-0.2, 1.1), xycoords='axes fraction', size=20, weight='bold', va='top')
ax2.annotate('E', xy=(-0.2, 1.27), xycoords='axes fraction', size=20, weight='bold', va='top')
ax3.annotate('F', xy=(-0.2, 1.27), xycoords='axes fraction', size=20, weight='bold', va='top')
ax4.annotate('G', xy=(-2, 1.1), xycoords='axes fraction', size=20, weight='bold', va='top')
ax6.annotate('H', xy=(-0.2, 1.1), xycoords='axes fraction', size=20, weight='bold', va='top')
ax7.annotate('J', xy=(-0.2, 1.27), xycoords='axes fraction', size=20, weight='bold', va='top')
ax8.annotate('K', xy=(-0.2, 1.27), xycoords='axes fraction', size=20, weight='bold', va='top')
ax9.annotate('L', xy=(-2, 1.1), xycoords='axes fraction', size=20, weight='bold', va='top')

ax9.set_facecolor('beige')
ax10.set_facecolor('beige')
ax4.set_facecolor('beige')
ax5.set_facecolor('beige')

#creating cut-out slanted lines for broken axis
d = 0.5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle='none', color='k', mew=1, clip_on=False)
ax9.plot([1, 1], [1, 0], transform=ax9.transAxes, **kwargs)
ax10.plot([0, 0], [1, 0], transform=ax10.transAxes, **kwargs)
ax4.plot([1, 1], [1, 0], transform=ax4.transAxes, **kwargs)
ax5.plot([0, 0], [1, 0], transform=ax5.transAxes, **kwargs)

#shifting rightmost subplots to the left to close gap created by broken axis
box = ax5.get_position()
print(box)
box.x0 -= 0.05
box.x1 -= 0.05
print(box)
ax5.set_position(box)

box = ax10.get_position()
print(box)
box.x0 -= 0.05
box.x1 -= 0.05
print(box)
ax10.set_position(box)

# Save plot
plt.savefig('outputs/plot.pdf')