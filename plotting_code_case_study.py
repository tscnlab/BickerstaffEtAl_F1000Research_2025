# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:57:07 2022
Modified on Thu Feb 8 12:16 2024

Python 3.11.5 (Anaconda)

@author: lbickerstaff
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy
from math import nan
import statistics as stat
#import ptitprince as pt
import seaborn as sns
import random
from sklearn.linear_model import LinearRegression
from brokenaxes import brokenaxes

plt.rcParams['font.family'] = ['Arial']

class plotPupilData:
    
    def allTrialsOneShotExp(conf = 0.7, moving_average_window = 40):
    
        """
        Function to:
            - Find timestamp of light onset from annotations.csv
            - Use the timestamp as start for pupil data plotting
        
        # conf = int. Confidence value betwen 0 and 1
        # trial = str. Format 'XXX'
        """
        
        random.seed(4)
        
        # Specifying path
        path = '/data/one_shot_exp/'
        dirs = os.listdir(path)
        dirs.pop(25) # minus trial 25 which doesn't have sneeze data for some reason
        dirs.pop(52) # minus trial 53 which doesn't have sneeze data for some reason
        
        # Preparing figure for plotting
        plt.figure(figsize=(10, 5))
        
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
        
        cmap = plt.get_cmap('Blues')
        
        # Plotting
        #plt.plot(average_data['60 mean'], label = '440 lux, N = '+str(len(data_60.columns)))
        #plt.plot(average_data['150 mean'], label = '1099 lux, N = '+str(len(data_150.columns)))
        #plt.plot(average_data['600 mean'], label = '4396 lux, N = '+str(len(data_600.columns)))
        #plt.plot(average_data['2400 mean'], label = '17582 lux, N = '+str(len(data_2400.columns)))
        average_data['60 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.4), label = '440 lux, N = '+str(len(data_60.columns)))
        average_data['150 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.6), label = '1099 lux, N = '+str(len(data_150.columns)))
        average_data['600 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.8), label = '4396 lux, N = '+str(len(data_600.columns)))
        average_data['2400 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.99), label = '17582 lux, N = '+str(len(data_2400.columns)))
        
        # plot parameters
        plt.xticks(ticks = [0, 
        round(len(average_data.index)/5), 
        round(len(average_data.index)/5)*2,
        round(len(average_data.index)/5)*3,
        round(len(average_data.index)/5)*4,
        len(average_data.index)], labels = ['-1', '-0.5', '0', '0.5', '1', '1.5'])
        plt.xlabel('Recording duration (min)', fontsize=12)
        plt.ylabel('Pupil diameter (mm)', fontsize=12)
        plt.xlim(left=0, right=len(average_data.index))
        plt.ylim(bottom=0, top=6)
        plt.vlines(round(len(average_data.index)/5)*2, ymin=0, ymax=10, colors='r', label='Light onset')
        plt.tick_params(right=True)
        plt.grid(axis = 'both', which = 'major', ls = '--')

        plt.title('Pupil data - One-shot experiment\nConfidence threshold = '+str(conf)+' - Moving average window = '+str(moving_average_window)+'\nN = '+str(len(dirs)))
        plt.legend()

    def allTrialsOneShotExpAverage(conf = 0.7):
        
        """
        Function to:
            - Find timestamp of light onset from annotations.csv
            - Use the timestamp as start for pupil data plotting
            - Average pupil diameter values during light sitmuli for each of the four illuminance values
            - Average pupil diameter values during dark period
        
        # conf = int. Confidence value betwen 0 and 1
        # trial = str. Format 'XXX'
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/one_shot_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        dirs.pop(25) # minus trial 25 which doesn't have sneeze data for some reason
        dirs.pop(52) # minus trial 53 which doesn't have sneeze data for some reason
        
        # Preparing figure for plotting
        #plt.figure(figsize=(10, 5))
        
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
        #plt = brokenaxes(xlims=((1.00000000e+00, 2.00000000e+00), (300, 30000)), hspace=.05, xscale='log')
        #plt.scatter(intensities, average, c = 'k')

        # Plot parameters
        #plt.errorbar(intensities, average, std, ecolor='k', capsize=4, fmt='none')
        #plt.xscale('log')
        #plt.ylim(bottom=0, top=6)
        #plt.xlim(left = 0.0, right=30000)
        #plt.xlabel('Intensity setting (lux)', fontsize=12)
        #plt.ylabel('Average pupil diameter (mm)', fontsize=12)
        #plt.xticks([1.00000000e+00, 300, 1000, 3000, 10000, 30000], labels=[0, 300, 1000, 3000, 10000, 30000])
        #plt.grid(axis = 'x', which = 'minor', ls = '--')
        #plt.grid(axis = 'x', which = 'major', ls = '-')
        #plt.title('Pupil data\nConfidence threshold = '+str(conf)+'\nmean +-STD, N = '+str(len(dirs)))
        
        ###################
        plt.figure(figsize=(6, 5))
        bax = brokenaxes(xlims=((-1000, 1000), (300, 30000)))
        bax.last_row[0].scatter(intensities[0], average[0], c='k')
        bax.last_row[0].errorbar(intensities[0], average[0], std[0], ecolor='k', capsize=4, fmt='none')
        bax.last_row[0].set_ylim(bottom = 0, top = 6)
        bax.last_row[0].set_yticks([0, 1, 2, 3, 4, 5, 6])
        bax.last_row[0].hlines(6, -1000, 30000, colors='k')
        bax.last_row[0].grid(axis = 'both', which = 'both', ls = '--')
        bax.last_row[0].annotate('N = '+n_labels[0], (intensities[0]+0.3, average[0]+0.3))
        bax.last_row[1].scatter(intensities_reg, average_reg, c='k')
        bax.last_row[1].errorbar(intensities, average, std, ecolor='k', capsize=4, fmt='none')
        bax.last_row[1].plot(xrange, y_pred, color='r', label='Slope = '+str(round(model.coef_[0], 2)), linestyle='dashed')
        bax.last_row[1].set_xscale('log')
        bax.last_row[1].set_ylim(bottom = 0, top = 6)
        bax.last_row[1].set_xticks([300, 1000, 3000, 10000, 30000], labels = [300, 1000, 3000, 10000, 30000])
        bax.last_row[1].grid(axis = 'both', which = 'both', ls = '--')
        bax.last_row[1].set_yticks([0, 1, 2, 3, 4, 5, 6])
        bax.last_row[1].tick_params(right=True)
        bax.last_row[1].vlines(30000, 0, 6, colors='k')
        bax.last_row[1].hlines(6, 0, 30000, colors='k')
        for i in range(4):
            bax.last_row[1].annotate('N = '+n_labels[i+1], (intensities[i+1]+intensities[i+1]/10, average[i+1]+0.3))
        bax.last_row[1].legend()
        bax.set_xlabel('Intensity setting (lux)', fontsize=12)
        bax.set_ylabel('Pupil diameter (mm)', fontsize=12)
        bax.set_title('Pupil data - One-shot experiment\nConfidence threshold = '+str(conf)+'\nMean +- STD, N = '+str(len(dirs)))
        
    def allTrialsDistExp(conf = 0.7, moving_average_window = 100):
        
        """
        Function to:
            - Find timestamp of light onset from annotations.csv
            - Use the timestamps as start for pupil data plotting
            
        # conf = int. Confidence value betwen 0 and 1
        # trial = str. Format 'XXX'
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/dist_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
        # Preparing figure for plotting
        plt.figure(figsize=(10, 5))
        
        data_2 = pd.DataFrame()
        data_4 = pd.DataFrame()
        data_6 = pd.DataFrame()
        
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
            
            if (info['distance'] == 2)[0]:
                data_2 = pd.concat([data_2, data['diameter_3d']], axis = 1)
            elif (info['distance'] == 4)[0]:
                data_4 = pd.concat([data_4, data['diameter_3d']], axis = 1)
            elif (info['distance'] == 6)[0]:
                data_6 = pd.concat([data_6, data['diameter_3d']], axis = 1)
            else:
                print('error')
                
            print(str(num)+'/'+str(len(dirs)))
            
            # Time of light onset, in seconds
            #onset_timestamp = (annotations.timestamp[0]/60)-min(data['pupil_timestamp'])
            #print(onset_timestamp)
            #print(data['pupil_timestamp'])
        
        # Filling DataFrame
        average_data = pd.DataFrame()
        average_data['2 mean'] = data_2.mean(1)
        average_data['2 std'] = data_2.std(1)
        average_data['4 mean'] = data_4.mean(1)
        average_data['4 std'] = data_4.std(1)
        average_data['6 mean'] = data_6.mean(1)
        average_data['6 std'] = data_6.std(1)
        
        cmap = plt.get_cmap('Blues')
        
        # Plotting
        #plt.plot(average_data['60 mean'], label = '440 lux, N = '+str(len(data_60.columns)))
        #plt.plot(average_data['150 mean'], label = '1099 lux, N = '+str(len(data_150.columns)))
        #plt.plot(average_data['600 mean'], label = '4396 lux, N = '+str(len(data_600.columns)))
        #plt.plot(average_data['2400 mean'], label = '17582 lux, N = '+str(len(data_2400.columns)))
        average_data['2 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.4), label = '2 meters, N = '+str(len(data_2.columns)))
        average_data['4 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.6), label = '4 meters, N = '+str(len(data_4.columns)))
        average_data['6 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.8), label = '6 meters, N = '+str(len(data_6.columns)))
        
        # plot parameters
        plt.xticks(ticks = [0, 
        round(len(average_data.index)/5), 
        round(len(average_data.index)/5)*2,
        round(len(average_data.index)/5)*3,
        round(len(average_data.index)/5)*4,
        len(average_data.index)], labels = ['-1', '-0.5', '0', '0.5', '1', '1.5'])
        plt.xlabel('Recording duration (min)', fontsize=12)
        plt.ylabel('Pupil diameter (mm)', fontsize=12)
        plt.xlim(left=0, right=len(average_data.index))
        plt.ylim(bottom=0, top=6)
        plt.vlines(round(len(average_data.index)/5)*2, ymin=0, ymax=10, colors='r', label='Light onset')
        plt.tick_params(right=True)
        plt.grid(axis = 'both', which = 'major', ls = '--')

        plt.title('Pupil data - Distance experiment\nConfidence threshold = '+str(conf)+' - Moving average window = '+str(moving_average_window)+'\nN = '+str(len(dirs)))
        plt.legend(loc='lower left')
        
    def allTrialsDistExpAverage(conf = 0.7):
        
        """
        Function to:
            - Find timestamp of light onset from annotations.csv
            - Use the timestamps as start for pupil data plotting
            
        # conf = int. Confidence value betwen 0 and 1
        # trial = str. Format 'XXX'
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/dist_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
        # Preparing figure for plotting
        plt.figure(figsize=(6, 5))
        
        data_2 = pd.DataFrame()
        data_4 = pd.DataFrame()
        data_6 = pd.DataFrame()
        
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
            
            if (info['distance'] == 2)[0]:
                data_2 = pd.concat([data_2, data['diameter_3d']], axis = 1)
            elif (info['distance'] == 4)[0]:
                data_4 = pd.concat([data_4, data['diameter_3d']], axis = 1)
            elif (info['distance'] == 6)[0]:
                data_6 = pd.concat([data_6, data['diameter_3d']], axis = 1)
            else:
                print('error')
                
            print(str(num)+'/'+str(len(dirs)))
            
        # Average
        average = [data_2.mean().mean(), 
                    data_4.mean().mean(), 
                    data_6.mean().mean()]
        print('Average: '+str(average))
        # Std
        std = [data_2.stack().std(),
              data_4.stack().std(),
              data_6.stack().std()]
        print('STD: '+str(std))
       
        n_labels = [str(len(data_2.columns)), str(len(data_4.columns)), str(len(data_6.columns))]
        
        distances = [2, 4, 6]
        log_distances = [0, 10]
        
        # Linear regression
        x = np.array(distances).reshape((-1, 1))
        y = np.array(average)
        model = LinearRegression()
        model.fit(x, y)
        print(f"slope: {model.coef_}")
        y_pred = model.predict(np.array(log_distances).reshape((-1, 1)))
            
        # Plotting
        
        plt.scatter(distances, average, c = 'k')
        plt.plot(log_distances, y_pred, color='r', label='Slope = '+str(round(model.coef_[0], 2)), linestyle='dashed')
        for i in range(3):
            plt.annotate('N = '+n_labels[i], (distances[i]+0.3, average[i]+0.3))
        
        # Plot parameters
        plt.errorbar(distances, average, std, ecolor='k', capsize=4, fmt='none')
        plt.ylim(bottom=0, top=6)
        plt.xlim(left=1, right=7)
        plt.xlabel('Distance from integrating sphere', fontsize=12)
        plt.ylabel('Pupil diameter (mm)', fontsize=12)
        plt.xticks(distances, labels=['2 meters',
                                      '4 meters',
                                      '6 meters'])
        plt.yticks([0, 1, 2, 3, 4, 5, 6])
        plt.tick_params(right=True)
        plt.grid(axis = 'both', which = 'both', ls = '--')
        plt.legend()
        plt.title('Pupil data - Distance experiment\nConfidence threshold = '+str(conf)+'\nMean +- STD, N = '+str(len(dirs)))
        
    def allTrials30minExp(conf = 0.7, moving_average_window = 40):
        
        """
        Function to:
            - Find timestamp of light onset from annotations.csv
            - Use the timestamps as start for pupil data plotting
            
        # conf = int. Confidence value betwen 0 and 1
        # trial = str. Format 'XXX'
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/30_min_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
        # Preparing figure for plotting
        plt.figure(figsize=(10, 5))
        
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
        
        # Plotting
        #plt.plot(average_data['60 mean'], label = '440 lux, N = '+str(len(data_60.columns)))
        #plt.plot(average_data['150 mean'], label = '1099 lux, N = '+str(len(data_150.columns)))
        #plt.plot(average_data['600 mean'], label = '4396 lux, N = '+str(len(data_600.columns)))
        #plt.plot(average_data['2400 mean'], label = '17582 lux, N = '+str(len(data_2400.columns)))
        average_data['0 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.4), label = '0 lux, N = '+str(len(data_0.columns)))
        average_data['60 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.4), label = '440 lux, N = '+str(len(data_60.columns)))
        average_data['150 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.6), label = '1099 lux, N = '+str(len(data_150.columns)))
        average_data['600 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.8), label = '4396 lux, N = '+str(len(data_600.columns)))
        average_data['2400 mean'].rolling(window=moving_average_window, min_periods=2).mean().plot(color = cmap(0.99), label = '17582 lux, N = '+str(len(data_2400.columns)))
        
        # plot parameters
        plt.xticks(ticks = [0, 
        round(len(average_data.index)/6), 
        round(len(average_data.index)/6)*2,
        round(len(average_data.index)/6)*3,
        round(len(average_data.index)/6)*4,
        round(len(average_data.index)/6)*5,
        len(average_data.index)], labels = ['0', '5', '10', '15', '20', '25', '30'])
        plt.xlabel('Recording duration (sec)', fontsize=12)
        plt.ylabel('Pupil diameter (mm)', fontsize=12)
        plt.xlim(left=0, right=len(average_data.index))
        plt.ylim(bottom=0, top=6)
        plt.tick_params(right=True)
        plt.grid(axis = 'both', which = 'major', ls = '--')
        
        plt.title('Pupil data - 30 minute experiment\nConfidence threshold = '+str(conf)+' - Moving average window = '+str(moving_average_window)+'\nN = '+str(len(dirs)*n_trials))
        plt.legend()
        
    def allTrials30minExpAverage(conf = 0.7):
        
        """
        Function to:
            - Find timestamp of light onset from annotations.csv
            - Use the timestamps as start for pupil data plotting
            
        # conf = int. Confidence value betwen 0 and 1
        # trial = str. Format 'XXX'
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/30_min_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
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
        plt.figure(figsize=(6, 5))
        bax = brokenaxes(xlims=((-1000, 1000), (300, 30000)))
        bax.last_row[0].scatter(intensities[0], average[0], c='k')
        bax.last_row[0].errorbar(intensities[0], average[0], std[0], ecolor='k', capsize=4, fmt='none')
        bax.last_row[0].set_ylim(bottom = 0, top = 6)
        bax.last_row[0].set_yticks([0, 1, 2, 3, 4, 5, 6])
        bax.last_row[0].hlines(6, -1000, 30000, colors='k')
        bax.last_row[0].grid(axis = 'both', which = 'both', ls = '--')
        bax.last_row[0].annotate('N = '+n_labels[0], (intensities[0]+0.3, average[0]+0.3))
        bax.last_row[1].scatter(intensities_reg, average_reg, c='k')
        bax.last_row[1].errorbar(intensities, average, std, ecolor='k', capsize=4, fmt='none')
        bax.last_row[1].plot(xrange, y_pred, color='r', label='Slope = '+str(round(model.coef_[0], 2)), linestyle='dashed')
        bax.last_row[1].set_xscale('log')
        bax.last_row[1].set_ylim(bottom = 0, top = 6)
        bax.last_row[1].set_xticks([300, 1000, 3000, 10000, 30000], labels = [300, 1000, 3000, 10000, 30000])
        bax.last_row[1].set_yticks([0, 1, 2, 3, 4, 5, 6])
        bax.last_row[1].tick_params(right=True)
        bax.last_row[1].vlines(30000, 0, 6, colors='k')
        bax.last_row[1].hlines(6, 0, 30000, colors='k')
        bax.last_row[1].grid(axis = 'both', which = 'both', ls = '--')
        for i in range(4):
            bax.last_row[1].annotate('N = '+n_labels[i+1], (intensities[i+1]+intensities[i+1]/10, average[i+1]+0.3))
        bax.last_row[1].legend()
        bax.set_xlabel('Intensity setting (lux)', fontsize=12)
        bax.set_ylabel('Pupil diameter (mm)', fontsize=12)
        bax.set_title('Pupil data - 30 minute experiment\nConfidence threshold = '+str(conf)+'\nMean +- STD, N = '+str(len(dirs)*n_trials))
        
    def singleTrialOneShotExp(conf = 0.7, trial = '047'):
    
        """
        Function to:
            - Find timestamp of light onset from annotations.csv
            - Use the timestamp as start for pupil data plotting
            
        # conf = int. Confidence value betwen 0 and 1
        # trial = str. Format 'XXX'
        """
        
        random.seed(4)
        
        # Preparing figure for plotting
        plt.figure(figsize=(10, 5))
        
        # Opening csv files
        data = pd.read_csv('D:/photic_sneeze/data/one_shot_exp/'+trial+'/exports/000/pupil_positions.csv')
        annotations = pd.read_csv('D:/photic_sneeze/data/one_shot_exp/'+trial+'/exports/000/annotations.csv')
        
        # Apply confidence threshold on pupil diameter values
        data['diameter_3d'].mask(data['confidence']<conf, other = nan, inplace = True)
        
        # Stripping DataFrame from unecessary data. Converting timestamps in min and normalising to light onset = 0 min
        data = pd.DataFrame({'pupil_timestamp': (data['pupil_timestamp'])/60, 'diameter_3d': data['diameter_3d']})
        #print(data)
        
        # Time of light onset, in seconds
        #onset_timestamp = (annotations.timestamp[0]/60)-min(data['pupil_timestamp'])
        #print(onset_timestamp)
        #print(data['pupil_timestamp'])
        
        # Plotting
        plt.plot(data['pupil_timestamp']-annotations.timestamp[annotations['label']=='lights on'].to_list()[0]/60, data['diameter_3d'])
        
        # plot parameters
        plt.vlines(0, ymin=0, ymax=10, colors='r', label='Light onset')
        plt.xlim(left=-1, right=1.5)
        plt.ylim(bottom=0, top=10)
        plt.xlabel('Recording duration (min)', fontsize=12)
        plt.ylabel('Pupil diameter (mm)', fontsize=12)
        plt.title('Pupil data\nConfidence threshold = '+str(conf)+'\nTrial = '+trial)
        plt.legend()

class plotLightData:
            
    def allEventsAverage(chosen_n = 100, logscale = False):
    
       """
       Function to:
           - Find timestamp of sneeze events from sneeze log in Notion
           - Plot average illuminance 20 min before and after this timestamp
       """
       
       random.seed(4)

       # Figure for plotting
       if logscale:
           plt.figure(figsize=(10, 10))
       else:
           plt.figure(figsize=(10, 10))

       # Opening light data file, getting rid of useless first lines and creating a DataFrame
       #with open('D:/photic_sneeze/data/actiwatch_data/actiwatch_combined_data.txt') as old, open('lightdata.txt', 'w') as new:
           #lines = old.readlines()
           #new.writelines(lines[28:])
       
       # Opening combined light data file and creating a DataFrame
       data = pd.read_csv('/Volumes/SSD Lucien/FICHIERS/Université/Master/M1 2021-2022/Stage M1/data/actiwatch/actiwatch_combined_data.txt', sep=';')

       # Stripping DataFrame from unecessary data.
       data = pd.DataFrame({'DATE/TIME': data['DATE/TIME'], 'LIGHT': data['LIGHT']})

       # Sneeze event timestamps from exported Notion sneeze log
       date_times = pd.read_csv(r'/Volumes/SSD Lucien/FICHIERS/Université/Master/M1 2021-2022/Stage M1/data/actiwatch/date_times_sneeze_log_notion.csv', sep = ';')

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

       # Plotting

       # Plot average and shaded error bar
       plt.plot(np.arange(-20,20.5,0.5), lightdata_calc_sneeze['average'], color='green', label='Sneeze event logged')
       plt.fill_between(np.arange(-20,20.5,0.5), 
                        lightdata_calc_sneeze['average']-lightdata_calc_sneeze['std'], 
                        lightdata_calc_sneeze['average']+lightdata_calc_sneeze['std'],
                        facecolor='green',
                        color='green',
                        alpha=0.3)

       #############################

       """
           - Plot light data when there was no sneeze on top
       """

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

       # Plot average and shaded error bar
       plt.plot(np.arange(-20,20.5,0.5), lightdata_calc_no_sneeze['average'], color='grey', label='No sneeze event logged (reference)')
       plt.fill_between(np.arange(-20,20.5,0.5), 
                        lightdata_calc_no_sneeze['average']-lightdata_calc_no_sneeze['std'], 
                        lightdata_calc_no_sneeze['average']+lightdata_calc_no_sneeze['std'],
                        facecolor='grey',
                        color='grey',
                        alpha=0.3)

       # Plot parameters
       if logscale:
           plt.vlines(0, ymin=0, ymax=30000, colors='r', label='Sneeze event')
           plt.yscale('log')
           plt.ylim(bottom=2e1, top=30000)
           plt.xlim(left=-20, right=20)
           plt.tick_params(right=True)
           plt.xlabel('Time (min)', fontsize=15)
           plt.ylabel('Light intensity (lux)', fontsize=15)
           plt.grid(axis = 'both', which = 'major', ls = '--')
           plt.legend(loc='lower left', fontsize=15)
           #plt.title('Light data - from 12/07 to 10/08\nAverage light level over 40 minutes +- SEM, N = '+str(n_sneeze))
       else:
           plt.vlines(0, ymin=0, ymax=12500, colors='r', label='Sneeze event')
           plt.ylim(bottom=2e1, top=12500)
           plt.xlim(left=-20, right=20)
           plt.tick_params(right=True)
           plt.xlabel('Time (min)', fontsize=12)
           plt.ylabel('Light intensity (lux)', fontsize=12)
           plt.grid(axis = 'both', which = 'major', ls = '--')
           plt.legend(loc=[.05, .75])
           plt.title('Light data - from 12/07 to 10/08\nAverage light level over 40 minutes +- SEM, N = '+str(n_sneeze))
           
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
               plt.scatter(i/2-20, 12000, color='k')
        
    def allEventsContrast():
    
        """
        Function to:
            - Find timestamp of sneeze events from sneeze log in Notion
            - Use the timestamp as start for light data plotting
            - Calculate mean light levels pre-sneeze (-5 to -2 min) and at sneeze event (-1 to +1 min)
            - Plot categorically, to see step in light intensity before and at sneeze event
        """
        
        random.seed(4)
        
        # Opening combined light data file and creating a DataFrame
        data = pd.read_csv('/Volumes/SSD Lucien/FICHIERS/Université/Master/M1 2021-2022/Stage M1/data/actiwatch/actiwatch_combined_data.txt', sep=';')

        # Stripping DataFrame from unecessary data.
        data = pd.DataFrame({'DATE/TIME': data['DATE/TIME'], 'LIGHT': data['LIGHT']})

        # Sneeze event timestamps from exported Notion sneeze log
        date_times = pd.read_csv(r'/Volumes/SSD Lucien/FICHIERS/Université/Master/M1 2021-2022/Stage M1/data/actiwatch/date_times_sneeze_log_notion.csv', sep = ';')

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
        
        # Plotting
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3, 10))
        
        # Plot mean and shaded error bar
        ax0.plot(x, [lightdata_presneeze_mean, lightdata_atsneeze_mean], color='green', alpha=0.5)
        ax0.plot(x, [lightdata_presneeze_mean, lightdata_atsneeze_mean], 'ko')
        ax0.set_yticks([1e1, 1e2, 1e3, 1e4], labels=['10', '100', '1000', '10000'])
        ax0.set_yscale('log')
        ax0.set_ylabel('Light intensity (lux)', fontsize=15)
        ax0.tick_params(right=True, labelsize=15)
        ax0.grid(axis = 'y', which = 'major', ls = '--')
        #ax0.set_title('Individual raw mean values')
        
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
        
        #ax1 = pt.RainCloud(data = df2/100) #/100 to have contrast in %
        ax1 = sns.stripplot(data = df2/100, color='black')
        ax1 = sns.boxplot(data = df2/100, width=0.5, palette='Greens')
        ax1.set_ylabel('x Pre-sneeze\n-5 to -2 min', fontsize=15)
        #ax1.set_title('Contrast')
        ax1.set_yscale('log')
        ax1.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3], labels=['0.1x', '1x', '10x', '100x', '1000x'])
        ax1.tick_params(right=True, labelsize=15)
        ax1.grid(axis = 'y', which = 'major', ls = '--')
        
        #fig.suptitle('Light data - from 12/07 to 10/08\nPre-sneeze and at sneeze event, N = '
                     #+str(len(event_timestamps_list)))
        
        # Stats
        stats = scipy.stats.ttest_rel(lightdata_presneeze_mean, lightdata_atsneeze_mean)
        p_value = round(stats[1], 13)
        ax0.text(.5, 25, 'P = '+str(p_value), fontsize=15)
    
    def singleEvent(event_timestamp = '17/07/2022 13:03'):
    
        """
        Function to:
            - Choose one timestamp from sneeze log in Notion
            - Plot 1 single sneeze event (+- 20 minutes)
            
        # event_timestamp = str of date and time
        """
        
        random.seed(4)
        
        # Opening light data file, getting rid of useless first lines and creating a DataFrame
        with open('D:/photic_sneeze/data/Log_1667_20220719142850251.txt') as old, open('lightdata.txt', 'w') as new:
            lines = old.readlines()
            new.writelines(lines[28:])
            
        data = pd.read_csv('lightdata.txt', sep=';')
        
        # Stripping DataFrame from unecessary data.
        data = pd.DataFrame({'DATE/TIME': data['DATE/TIME'], 'LIGHT': data['LIGHT']})
        
        lightdata = pd.DataFrame()
        
        # Create a new dataframe, add the light values for the desired range for the chosen event
        index_value = data[data['DATE/TIME'] == event_timestamp+':25']['LIGHT'].index.tolist()[0]
        lightdata[0] = data[index_value-40:index_value+41]['LIGHT'].tolist()
        
        # Plotting
    
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(-20,20.5,0.5), lightdata)
        plt.ylim(bottom=0, top=20000)
        plt.xlim(left=-20, right=20)
        plt.xlabel('Time (min)', fontsize=12)
        plt.ylabel('Light intensity (lux)', fontsize=12)
        plt.title('Light data - '+event_timestamp)
        plt.vlines(0, ymin=0, ymax=20000, colors='r', label='Sneeze event')
        plt.legend()
        
class plotTickleData:
        
    def allTrialsOneShotExp(): # Not ideal as we visualise mean and std on discrete values
    
        """
        Function to:
            - Retrieve tickle ratings from sneeze_data_xxx.csv, for all trial folders
            - Plot them as a function of light intensity setting
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/one_shot_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        dirs.pop(25) # trial 025 failed. Pupil recording worked but not tickle rating...
        dirs.pop(52) # trial 053 failed. Pupil recording worked but not tickle rating...
                
        # Preparing figure for plotting
        plt.figure(figsize=(5, 5))
        
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
        plt.scatter(intensities, average, c = 'k')
        plt.plot(xrange, y_pred, color='r', label='Slope = '+str(round(model.coef_[0], 2)), linestyle='dashed')
        for i in range(4):
            plt.annotate('N = '+n_labels[i], (intensities[i]+intensities[i]/10, average[i]-0.3))
        
        # Plot parameters
        plt.errorbar(intensities, average, std, ecolor='k', capsize=4, fmt='none')
        plt.xscale('log')
        plt.ylim(bottom=0, top=10)
        plt.xlim(left=300, right=30000)
        plt.xlabel('Intensity setting (lux)', fontsize=12)
        plt.ylabel('Tickle rating /10', fontsize=12)
        plt.xticks([300, 1000, 3000, 10000, 30000], labels=[300, 1000, 3000, 10000, 30000])
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.tick_params(right=True)
        plt.title('Tickle sensation ratings - One-shot experiment\nMean +-STD, N = '+str(len(dirs)))
        plt.grid(axis = 'both', which = 'both', ls = '--')
        #plt.grid(axis = 'x', which = 'major', ls = '-')
        plt.legend()
        
    def allTrialsOneShotExpBarChart():
        
        """
        Function to:
            - Retrieve tickle ratings from sneeze_data_xxx.csv, for all trial folders
            - Plot them as a function of light intensity setting
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/one_shot_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
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
            
        cmap = plt.get_cmap('YlOrRd')
        
        # Preparing figure for plotting
        plt.figure(figsize=(5, len(listall)))
        
        # Plot parameters
        for i in range(len(listall)): # intensity
            n=0
            for j in range(len(listall[i])):
                plt.barh(intensities[i],
                    1/len(listall[i]),
                    left=n, 
                    color=cmap(listall[i][j]/10))
                n+=1/len(listall[i])
        
        for i in range(10): # Generating blank values for each level, with the corresponding label and color for the legend
            plt.barh(intensities[0],
                     0,
                     color=cmap(i/10),
                     label=str(i)+'/10')
                
        # Plot parameters
        plt.ylabel('Intensity setting (lux)', fontsize=12)
        plt.xlabel('Proportion', fontsize=12)
        plt.yticks(intensities, labels=['440\nN = '+str(len(list60)), 
                                        '1099\nN = '+str(len(list150)), 
                                        '4396\nN = '+str(len(list600)), 
                                        '17582\nN = '+str(len(list2400))])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.title('Tickle sensation ratings - One-shot experiment\nN = '+str(len(dirs)))
        plt.legend(loc=[1.03, 0.2])

    def allTrialsDistExp():
    
        """
        Function to:
            - Retrieve tickle ratings from sneeze_data_xxx.csv, for all trial folders
            - Plot them as a function of distance from integrating sphere
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/dist_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
        # Preparing figure for plotting
        plt.figure(figsize=(5, 5))
        
        list2 = []
        list4 = []
        list6 = []
        
        for i in dirs:
            
            # Opening csv file
            data = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
            
            if (data['distance'] == 2)[0]:
                #print(data['reponse_q2'][0])
                list2.append(int(data['reponse_q2'][0]))
            elif (data['distance'] == 4)[0]:
                #print(data['reponse_q2'][0])
                list4.append(int(data['reponse_q2'][0]))
            elif (data['distance'] == 6)[0]:
                #print(data['reponse_q2'][0])
                list6.append(int(data['reponse_q2'][0]))
            else:
                print('error')
            
        # Average
        average = [stat.mean(list2), stat.mean(list4), stat.mean(list6)]
        # Std
        std = [stat.stdev(list2), stat.stdev(list4), stat.stdev(list6)]
        
        n_labels = [str(len(list2)), str(len(list4)), str(len(list6))]
        
        distances = [2, 4, 6]
        log_distances = [0, 10]
        
        # Linear regression
        x = np.array(distances).reshape((-1, 1))
        y = np.array(average)
        model = LinearRegression()
        model.fit(x, y)
        print(f"slope: {model.coef_}")
        y_pred = model.predict(np.array(log_distances).reshape((-1, 1)))
            
        # Plotting
        
        plt.scatter(distances, average, c = 'k')
        plt.plot(log_distances, y_pred, color='r', label='Slope = '+str(round(model.coef_[0], 2)), linestyle='dashed')
        for i in range(3):
            plt.annotate('N = '+n_labels[i], (distances[i]+0.3, average[i]+0.3))
        
        # Plot parameters
        plt.errorbar(distances, average, std, ecolor='k', capsize=4, fmt='none')
        plt.ylim(bottom=0, top=10)
        plt.xlim(left=1, right=7)
        plt.xlabel('Distance from integrating sphere', fontsize=12)
        plt.ylabel('Tickle rating /10', fontsize=12)
        plt.xticks(distances, labels=['2 meters',
                                      '4 meters',
                                      '6 meters'])
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plt.tick_params(right=True)
        plt.grid(axis = 'both', which = 'both', ls = '--')
        plt.legend()
        plt.title('Tickle sensation ratings - Distance experiment\nMean +-STD, N = '+str(len(dirs)))
        
    def allTrialsDistExpBarChart():
        
        """
        Function to:
            - Retrieve tickle ratings from sneeze_data_xxx.csv, for all trial folders
            - Plot them as a function of light intensity setting
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/dist_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
        list2 = []
        list4 = []
        list6 = []
        
        for i in dirs:
            
            # Opening csv file
            data = pd.read_csv(path+str(i)+'/sneeze_data_'+str(i)+'.csv')
            
            if (data['distance'] == 2)[0]:
                #print(data['reponse_q2'][0])
                list2.append(int(data['reponse_q2'][0]))
            elif (data['distance'] == 4)[0]:
                #print(data['reponse_q2'][0])
                list4.append(int(data['reponse_q2'][0]))
            elif (data['distance'] == 6)[0]:
                #print(data['reponse_q2'][0])
                list6.append(int(data['reponse_q2'][0]))
            else:
                print('error')
        
        distances = ['2', '4', '6']

        listall = [list2, list4, list6]

        for i in listall: # sorting lists
            i.sort()
            
        cmap = plt.get_cmap('YlOrRd')
        
        # Preparing figure for plotting
        plt.figure(figsize=(5, len(listall)))
        
        # Plot parameters
        for i in range(len(listall)): # intensity
            n=0
            for j in range(len(listall[i])):
                plt.barh(distances[i], 
                        1/len(listall[i]), 
                        left=n, 
                        color=cmap(listall[i][j]/10))
                n+=1/len(listall[i])
                
        for i in range(10): # Generating blank values for each level, with the corresponding label and color for the legend
            plt.barh(distances[0],
                     0,
                     color=cmap(i/10),
                     label=str(i)+'/10')
        
        # Plot parameters
        plt.ylabel('Distance from integrating sphere', fontsize=12)
        plt.xlabel('Proportion', fontsize=12)
        plt.yticks(distances, labels=['2 meters\nN = '+str(len(list2)), 
                                      '4 meters\nN = '+str(len(list4)),
                                      '6 meters\nN = '+str(len(list6))])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.title('Tickle sensation ratings - Distance experiment\nN = '+str(len(dirs)))
        plt.legend(loc=[1.03, .05])
        
    def allTrials30minExp():
        
        """
        Function to:
            - Retrieve tickle ratings from sneeze_data_xxx.csv
            - Plot them as a function of time (?)
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/30_min_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
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
        #print(y_pred)
        
        n_labels = [str(len(list0)), str(len(list60)), str(len(list150)), str(len(list600)), str(len(list2400))]   
        
        # Plotting 
        # Plot parameters
        plt.figure(figsize=(5, 5))
        bax = brokenaxes(xlims=((-1000, 1000), (300, 30000))) 
        bax.last_row[0].scatter(intensities[0], average[0], c='k')
        bax.last_row[0].errorbar(intensities[0], average[0], std[0], ecolor='k', capsize=4, fmt='none')
        bax.last_row[0].set_ylim(bottom = 0, top = 10)
        bax.last_row[0].set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bax.last_row[0].hlines(10, -1000, 30000, colors='k')
        bax.last_row[0].grid(axis = 'both', which = 'both', ls = '--')
        bax.last_row[0].annotate('N = '+n_labels[0], (intensities[0]+0.3, average[0]+0.3))
        bax.last_row[1].scatter(intensities_reg, average_reg, c='k')
        bax.last_row[1].errorbar(intensities, average, std, ecolor='k', capsize=4, fmt='none')
        bax.last_row[1].plot(xrange, y_pred, color='r', label='Slope = '+str(round(model.coef_[0], 2)), linestyle='dashed')
        bax.last_row[1].set_xscale('log')
        bax.last_row[1].set_ylim(bottom = 0, top = 10)
        bax.last_row[1].set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bax.last_row[1].tick_params(right=True)
        bax.last_row[1].set_xticks([300, 1000, 3000, 10000, 30000], labels = [300, 1000, 3000, 10000, 30000])
        bax.last_row[1].vlines(30000, 0, 10, colors='k')
        bax.last_row[1].hlines(10, 0, 30000, colors='k')
        bax.last_row[1].grid(axis = 'both', which = 'both', ls = '--')
        for i in range(4):
            bax.last_row[1].annotate('N = '+n_labels[i+1], (intensities[i+1]+intensities[i+1]/10, average[i+1]+0.3))
        bax.last_row[1].legend()
        bax.set_xlabel('Intensity setting (lux)', fontsize=12)
        bax.set_ylabel('Tickle rating /10', fontsize=12)
        bax.set_title('Tickle sensation ratings - 30 minute experiment\nMean +-STD, N = '+str(len(dirs)*n_trials))
        
    def allTrials30minExpBarChart():
        
        """
        Function to:
            - Retrieve tickle ratings from sneeze_data_xxx.csv
            - Plot them as a function of time (?)
        """
        
        random.seed(4)
        
        # Specifying path
        path = 'D:/photic_sneeze/data/30_min_exp/'
        dirs = os.listdir(path)[:-1] # minus test directory
        
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
            
        cmap = plt.get_cmap('YlOrRd')
        
        # Preparing figure for plotting
        plt.figure(figsize=(5, len(listall)))
        
        # Plot parameters
        for i in range(len(listall)): # intensity
            n=0
            for j in range(len(listall[i])):
                plt.barh(intensities[i], 
                        1/len(listall[i]), 
                        left=n, 
                        color=cmap(listall[i][j]/10))
                n+=1/len(listall[i])
                
        for i in range(10): # Generating blank values for each level, with the corresponding label and color for the legend
            plt.barh(intensities[0],
                     0,
                     color=cmap(i/10),
                     label=str(i)+'/10')
        
        # Plot parameters
        plt.ylabel('Intensity setting (lux)', fontsize=12)
        plt.xlabel('Proportion', fontsize=12)
        plt.yticks(intensities, labels=['0\nN = '+str(len(list0)),
                                        '440\nN = '+str(len(list60)), 
                                        '1099\nN = '+str(len(list150)), 
                                        '4396\nN = '+str(len(list600)), 
                                        '17582\nN = '+str(len(list2400))])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.title('Tickle sensation ratings - 30 minute experiment\nN = '+str(len(dirs)*n_trials))
        plt.legend(loc=[1.03, 0.3])
        
#plotPupilData.allTrialsOneShotExp(conf = 0.7, moving_average_window=40)
#plotPupilData.allTrialsOneShotExpAverage(conf = 0.7)
#plotPupilData.allTrialsDistExp(conf = 0.7, moving_average_window=40)
#plotPupilData.allTrialsDistExpAverage(conf = 0.7)
#plotPupilData.allTrials30minExp(conf = 0.7, moving_average_window=40)
#plotPupilData.allTrials30minExpAverage(conf = 0.7)
#plotPupilData.singleTrialOneShotExp(conf = 0.7, trial = '055')

#plotLightData.allEventsAverage(chosen_n = 82, logscale = True)
#plotLightData.allEventsAverage(chosen_n = 82, logscale = False)
plotLightData.allEventsContrast()
#plotLightData.singleEvent()

#plotTickleData.allTrialsOneShotExp()
#plotTickleData.allTrialsOneShotExpBarChart()
#plotTickleData.allTrialsDistExp()
#plotTickleData.allTrialsDistExpBarChart()
#plotTickleData.allTrials30minExp()
#plotTickleData.allTrials30minExpBarChart()