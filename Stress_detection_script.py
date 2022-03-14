
#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Stress_detection_script.py
@Time    :   2022/03/17 09:45:59
@Author  :   Sri Charan 
@Contact :   dgscharan@gmail.com
'''

import os
import numpy as np
import pandas as pd
import zipfile
import fnmatch
import datetime
import logging
import flirt.reader.empatica
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxopt as cv
from neurokit2 import eda_phasic
import multiprocessing

# rootPath = r"./"
# pattern = '*.zip'

rootPath = input("Enter Folder Path : ")
pattern = input("Enter File Name : ")
for root, dirs, files in os.walk(rootPath):
    for filename in fnmatch.filter(files, pattern):
        print(os.path.join(root, filename))
        zipfile.ZipFile(os.path.join(root, filename)).extractall(
            os.path.join(root, os.path.splitext(filename)[0]))
dir = os.path.splitext(pattern)[0]

# os.listdir(dir)


class process:
    def __init__(self) -> None:
        pass

    def moving_avarage_smoothing(X, k, description_str):
        S = np.zeros(X.shape[0])
        for t in tqdm(range(X.shape[0]), desc=description_str):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
        return S

    def deviation_above_mean(unit, mean_unit, std_unit):
        if unit == 0:
            return (mean_unit)
        else:
            return (mean_unit + (unit*std_unit))

    def Starting_timeStamp(column, time_frames, deviation_metric):
        starting_time_index = []
        for i in range(len(column)-1):
            if column[i] < deviation_metric and column[i+1] > deviation_metric:
                starting_time_index.append(time_frames[i])
        return starting_time_index

    def Ending_timeStamp(column, time_frames, deviation_metric):
        time_index = []
        for i in range(len(column)-1):
            if column[i] > deviation_metric and column[i+1] < deviation_metric:
                time_index.append(time_frames[i])
        if column[len(column) - 1] > deviation_metric:
            time_index.insert(
                len(time_index), time_frames[len(time_frames) - 1])
        else:
            pass
        return time_index


def Extract_HRV_Information():

    global hrv_features  # declaring global to get access them for combined plot function
    global hrv_events_df  # declaring global to get access them for combined plot function

    ibi = pd.read_csv(rootPath+'/'+dir+'\IBI.csv')
    mean_ibi = ibi[' IBI'].mean()
    average_heart_rate = 60/mean_ibi
    print('mean ibi is :', mean_ibi)
    print('mean heart rate :', average_heart_rate.round())
    ibis = flirt.reader.empatica.read_ibi_file_into_df(
        rootPath+'/'+dir + '\IBI.csv')
    hrv_features = flirt.get_hrv_features(
        ibis['ibi'], 128, 1, ["td", "fd"], 0.2)
    hrv_features = hrv_features.dropna(how='any', axis=0)
    hrv_features.reset_index(inplace=True)

    # smoothing the curve
    print('\n', '******************** Smoothing The Curve ********************', '\n')
    MAG_K500 = process.moving_avarage_smoothing(
        hrv_features['hrv_rmssd'], 500, "Processing HRV Data")

    hrv_features['MAG_K500'] = MAG_K500
    # hrv_features.to_csv("./Metadata/"+ dir+"_HRV.csv")
    # hrv_features.to_csv(os.path.join('./Metadata'+dir+'_HRV.csv'))
    mean_rmssd = hrv_features['hrv_rmssd'].mean()
    std_rmssd = hrv_features['hrv_rmssd'].std()

    # getting the starrting and ending time of of the signal
    starting_timestamp = process.Starting_timeStamp(hrv_features['MAG_K500'], hrv_features['datetime'],
                                                    process.deviation_above_mean(1, mean_rmssd, std_rmssd))
    ending_timestamp = process.Ending_timeStamp(hrv_features['MAG_K500'], hrv_features['datetime'],
                                                process.deviation_above_mean(1, mean_rmssd, std_rmssd))

    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30, 10))
        ax2.plot(hrv_features['datetime'],
                 hrv_features['MAG_K500'], color='red')
        fig.savefig('./Plots/HRV_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        else:
            pass
        difference = []  # initialization of result list
        time_delta_minutes = []
        desired_time_index = []
        zip_object = zip(ending_timestamp, starting_timestamp)
        for list1_i, list2_i in zip_object:
            # append each difference to list
            difference.append(list1_i-list2_i)
        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)
        time_delta_minutes
        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 5.00:
                desired_time_index.append(i)
        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        hrv_events_df = pd.concat(frames,  axis=1)
        hrv_events_df.columns = ['Starting Timestamp', 'Ending Timestamp']
        hrv_events_df = hrv_events_df.loc[desired_time_index, :]
        
        # logic to combine smaller intervals of 10 mins 
        try:
            for i in range(len(hrv_events_df['Starting Timestamp'])):
                # print(hrv_events_df['Starting Timestamp'][i+1])
                for j in range(len(hrv_events_df['Ending Timestamp'])):
                    if ((hrv_events_df['Starting Timestamp'][i+1] - hrv_events_df['Ending Timestamp'][i]) < datetime.timedelta(minutes=10) and 
                        (hrv_events_df['Starting Timestamp'][i+1] - hrv_events_df['Ending Timestamp'][i]) > datetime.timedelta(minutes=0, seconds=0)):
                        hrv_events_df['Starting Timestamp'][i+1] = hrv_events_df['Ending Timestamp'][i]
                        # print(hrv_events_df['Starting Timestamp'][i+1], '\n', hrv_events_df['Ending Timestamp'][i])    
                    else:
                        pass     
        except KeyError as error:
            logging.info('range error in delta time adjustments')
        
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(hrv_features['datetime'],
                hrv_features['MAG_K500'], color='red')
        for d in hrv_events_df.index:
            ax.axvspan(hrv_events_df['Starting Timestamp'][d], hrv_events_df['Ending Timestamp']
                       [d], facecolor="g", edgecolor="none", alpha=0.5)
        ax.relim()
        ax.autoscale_view()
        fig.savefig('./Plots/HRV_figure.png')
        return hrv_features, hrv_events_df


def Extract_ACC_Infromation():
    global acc_df
    global acc_events_df

    acc_df = pd.read_csv(rootPath+'/'+dir + '/ACC.csv')
    acc_df = flirt.reader.empatica.read_acc_file_into_df(
        rootPath+'/'+dir + '/ACC.csv')
    acc_df['Magnitude'] = np.sqrt(
        acc_df['acc_x']**2 + acc_df['acc_y']**2 + acc_df['acc_z']**2)
    print("Magnitude Mean : ", acc_df['Magnitude'].mean())
    acc_df.reset_index(inplace=True)

    print('\n', '******************** Smoothing The ACC Curve ********************', '\n')
    MAG_K500 = process.moving_avarage_smoothing(
        acc_df['Magnitude'], 15000, "Processing ACC Data")

    acc_df['MAG_K500'] = MAG_K500
    # acc_df.to_csv("./Metadata/"+ dir+"_ACC.csv")
    mean_acc_magnitude = acc_df['Magnitude'].mean()
    std_acc_magnitude = acc_df['Magnitude'].std()
    print("Average Magnitude of the Acc Data : ", mean_acc_magnitude)

    starting_timestamp = process.Starting_timeStamp(acc_df['MAG_K500'], acc_df['datetime'],
                                                    process.deviation_above_mean(0.20, mean_acc_magnitude, std_acc_magnitude))

    ending_timestamp = process.Ending_timeStamp(acc_df['MAG_K500'], acc_df['datetime'],
                                                process.deviation_above_mean(0.20, mean_acc_magnitude, std_acc_magnitude))

    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30, 10))
        ax2.plot(acc_df['datetime'], acc_df['MAG_K500'], color='red')
        fig.savefig('./Plots/ACC_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        difference = []  # initialization of result list
        time_delta_minutes = []
        desired_time_index = []
        zip_object = zip(ending_timestamp, starting_timestamp)

        for list1_i, list2_i in zip_object:
            # append each difference to list
            difference.append(list1_i-list2_i)

        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)

        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 2.00:
                desired_time_index.append(i)

        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        acc_events_df = pd.concat(frames,  axis=1)
        acc_events_df.columns = ['Starting Timestamp', 'Ending Timestamp']
        acc_events_df = acc_events_df.loc[desired_time_index, :]
        # acc_events_df.to_csv(rootPath+"timestamp_" +dir+ "_ACC.csv")
        
        try:
            for i in range(len(acc_events_df['Starting Timestamp'])):
                # print(acc_events_df['Starting Timestamp'][i+1])
                for j in range(len(acc_events_df['Ending Timestamp'])):
                    if ((acc_events_df['Starting Timestamp'][i+1] - acc_events_df['Ending Timestamp'][i]) < datetime.timedelta(minutes=10) and 
                        (acc_events_df['Starting Timestamp'][i+1] - acc_events_df['Ending Timestamp'][i]) > datetime.timedelta(minutes=0, seconds=0)):
                        acc_events_df['Starting Timestamp'][i+1] = acc_events_df['Ending Timestamp'][i]
                        # print(acc_events_df['Starting Timestamp'][i+1], '\n', acc_events_df['Ending Timestamp'][i])    
                    else:
                        pass     
        except KeyError as error:
            logging.info('range error in delta time adjustments')
        
        acc_events_df
        
        fig, ax2 = plt.subplots(figsize=(30, 10))
        ax2.plot(acc_df['datetime'], acc_df['MAG_K500'], color='red')
        for d in acc_events_df.index:
            ax2.axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp']
                        [d], facecolor="g", edgecolor="none", alpha=0.5)
        ax2.relim()
        ax2.autoscale_view()
        fig.savefig('./Plots/ACC_figure.png')


def Extract_GSR_Phasic_Information():
    global eda_df
    global eda_phasic_df
    global eda_phasic_events_df

    eda_df = pd.read_csv(rootPath+'/'+dir+'/EDA.csv')
    eda_df = flirt.reader.empatica.read_eda_file_into_df(
        rootPath+'/' + dir + '/EDA.csv')
    eda_df.reset_index(inplace=True)
    eda = np.array(eda_df['eda'])
    Phasic_Tonic_DF = eda_phasic(eda, 4, method='cvxEDA')
    eda_df['tonic'] = Phasic_Tonic_DF['EDA_Tonic']
    eda_df['phasic'] = Phasic_Tonic_DF['EDA_Phasic']
    eda_phasic_df = eda_df.copy()

    print('\n', '******************** Smoothing The EDA Phasic Curve ********************', '\n')
    MAG_K500 = process.moving_avarage_smoothing(
        eda_phasic_df['phasic'], 2000, "Processing EDA Phasic Data")

    eda_phasic_df['MAG_K500'] = MAG_K500
    # hrv_features.to_csv('hrv_features.csv')
    mean_eda_phasic = eda_phasic_df['phasic'].mean()
    std_eda_phasic = eda_phasic_df['phasic'].std()

    starting_timestamp = process.Starting_timeStamp(eda_phasic_df['MAG_K500'], eda_phasic_df['datetime'],
                                                    process.deviation_above_mean(1, mean_eda_phasic, std_eda_phasic))

    ending_timestamp = process.Ending_timeStamp(eda_phasic_df['MAG_K500'], eda_phasic_df['datetime'],
                                                process.deviation_above_mean(1, mean_eda_phasic, std_eda_phasic))

    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30, 10))
        ax2.plot(eda_phasic_df['datetime'],
                 eda_phasic_df['MAG_K500'], color='red')
        fig.savefig('./Plots/EDA_Phasic_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        difference = []  # initialization of result list
        time_delta_minutes = []
        desired_time_index = []
        zip_object = zip(ending_timestamp, starting_timestamp)
        for list1_i, list2_i in zip_object:
            # append each difference to list
            difference.append(list1_i-list2_i)
        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)
        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 2.00:
                desired_time_index.append(i)
        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        eda_phasic_events_df = pd.concat(frames,  axis=1)
        eda_phasic_events_df.columns = [
            'Starting Timestamp', 'Ending Timestamp']
        eda_phasic_events_df = eda_phasic_events_df.loc[desired_time_index, :]
        # eda_phasic_events_df.to_csv(rootPath+"timestamp_" + dir + "_EDA.csv")
        
        try:
            for i in range(len(eda_phasic_events_df['Starting Timestamp'])):
                # print(eda_phasic_events_df['Starting Timestamp'][i+1])
                for j in range(len(eda_phasic_events_df['Ending Timestamp'])):
                    if ((eda_phasic_events_df['Starting Timestamp'][i+1] - eda_phasic_events_df['Ending Timestamp'][i]) < datetime.timedelta(minutes=10) and 
                        (eda_phasic_events_df['Starting Timestamp'][i+1] - eda_phasic_events_df['Ending Timestamp'][i]) > datetime.timedelta(minutes=0, seconds=0)):
                        eda_phasic_events_df['Starting Timestamp'][i+1] = eda_phasic_events_df['Ending Timestamp'][i]
                        # print(eda_phasic_events_df['Starting Timestamp'][i+1], '\n', eda_phasic_events_df['Ending Timestamp'][i])    
                    else:
                        pass     
        except KeyError as error:
            logging.info('range error in delta time adjustments')
        
        fig, ax3 = plt.subplots(figsize=(30, 10))
        ax3.plot(eda_phasic_df['datetime'],
                 eda_phasic_df['MAG_K500'], color='red')
        for d in eda_phasic_events_df.index:
            ax3.axvspan(eda_phasic_events_df['Starting Timestamp'][d],
                        eda_phasic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        ax3.relim()
        ax3.autoscale_view()
        fig.savefig('./Plots/EDA_Phasic_figure.png')
        return eda_df


def Extract_GSR_Tonic_Information():
    global eda_tonic_df
    global eda_tonic_events_df

    eda_df = pd.read_csv(rootPath+'/'+dir+'/EDA.csv')
    eda_df = flirt.reader.empatica.read_eda_file_into_df(
        rootPath+'/' + dir + '/EDA.csv')
    eda_df.reset_index(inplace=True)
    eda = np.array(eda_df['eda'])
    Phasic_Tonic_DF = eda_phasic(eda, 4, method='cvxEDA')
    eda_df['tonic'] = Phasic_Tonic_DF['EDA_Tonic']
    eda_df['phasic'] = Phasic_Tonic_DF['EDA_Phasic']
    eda_tonic_df = eda_df.copy()

    print('\n', '******************** Smoothing The EDA Tonic Curve ********************', '\n')
    MAG_K500 = process.moving_avarage_smoothing(
        eda_tonic_df['tonic'], 2000, "Processing EDA Tonic Data")
    eda_tonic_df['MAG_K500'] = MAG_K500
    # hrv_features.to_csv('hrv_features.csv')
    mean_eda_tonic = eda_tonic_df['tonic'].mean()
    std_eda_tonic = eda_tonic_df['tonic'].std()

    starting_timestamp = process.Starting_timeStamp(eda_tonic_df['MAG_K500'], eda_tonic_df['datetime'],
                                                    process.deviation_above_mean(1, mean_eda_tonic, std_eda_tonic))

    ending_timestamp = process.Ending_timeStamp(eda_tonic_df['MAG_K500'], eda_tonic_df['datetime'],
                                                process.deviation_above_mean(1, mean_eda_tonic, std_eda_tonic))

    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30, 10))
        ax2.plot(eda_tonic_df['datetime'],
                 eda_tonic_df['MAG_K500'], color='red')
        fig.savefig('./Plots/EDA_Phasic_figure.png')
    else:
        print("entering final else block")
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        difference = []  # initialization of result list
        time_delta_minutes = []
        desired_time_index = []
        zip_object = zip(ending_timestamp, starting_timestamp)
        for list1_i, list2_i in zip_object:
            # append each difference to list
            difference.append(list1_i-list2_i)
        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)
        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 2.00:
                desired_time_index.append(i)
        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        eda_tonic_events_df = pd.concat(frames,  axis=1)
        eda_tonic_events_df.columns = [
            'Starting Timestamp', 'Ending Timestamp']
        eda_tonic_events_df = eda_tonic_events_df.loc[desired_time_index, :]
        # eda_tonic_events_df.to_csv(rootPath+"timestamp_" +dir+ "_EDA.csv")
        
        try:
            for i in range(len(eda_tonic_events_df['Starting Timestamp'])):
                # print(eda_tonic_events_df['Starting Timestamp'][i+1])
                for j in range(len(eda_tonic_events_df['Ending Timestamp'])):
                    if ((eda_tonic_events_df['Starting Timestamp'][i+1] - eda_tonic_events_df['Ending Timestamp'][i]) < datetime.timedelta(minutes=10) and 
                        (eda_tonic_events_df['Starting Timestamp'][i+1] - eda_tonic_events_df['Ending Timestamp'][i]) > datetime.timedelta(minutes=0, seconds=0)):
                        eda_tonic_events_df['Starting Timestamp'][i+1] = eda_tonic_events_df['Ending Timestamp'][i]
                        # print(eda_tonic_events_df['Starting Timestamp'][i+1], '\n', eda_tonic_events_df['Ending Timestamp'][i])    
                    else:
                        pass     
        except KeyError as error:
            logging.info('range error in delta time adjustments')
        
        fig, ax4 = plt.subplots(figsize=(30, 10))
        ax4.plot(eda_tonic_df['datetime'],
                 eda_tonic_df['MAG_K500'], color='red')
        for d in eda_tonic_events_df.index:
            ax4.axvspan(eda_tonic_events_df['Starting Timestamp'][d],
                        eda_tonic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        ax4.relim()
        ax4.autoscale_view()
        fig.savefig('./Plots/EDA_tonic_figure.png')


def Extract_Heart_Rate_Features():
    global hr_df
    global hr_events_df

    hr_df = flirt.reader.empatica.read_hr_file_into_df(
        rootPath+'/'+dir+'/HR.csv')
    hr_df.reset_index(inplace=True)

    print('\n', '******************** Smoothing The Heart Rate Curve ********************', '\n')
    MAG_K500 = process.moving_avarage_smoothing(
        hr_df['hr'], 500, "Processing Heart Rate Data")

    hr_df['MAG_K500'] = MAG_K500
    # hrv_features.to_csv('hrv_features.csv')
    hr_avg = hr_df['MAG_K500'].mean()
    hr_std = hr_df['MAG_K500'].std()

    starting_timestamp = process.Starting_timeStamp(
        hr_df['MAG_K500'], hr_df['datetime'], process.deviation_above_mean(1, hr_avg, hr_std))
    ending_timestamp = process.Ending_timeStamp(
        hr_df['MAG_K500'], hr_df['datetime'], process.deviation_above_mean(1, hr_avg, hr_std))

    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30, 10))
        ax2.plot(hr_df['datetime'], hr_df['MAG_K500'], color='red')
        fig.savefig('./Plots/Heart_rate_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        difference = []  # initialization of result list
        time_delta_minutes = []
        desired_time_index = []
        zip_object = zip(ending_timestamp, starting_timestamp)
        for list1_i, list2_i in zip_object:
            # append each difference to list
            difference.append(list1_i-list2_i)
        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)
        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 2.00:
                desired_time_index.append(i)
        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        hr_events_df = pd.concat(frames,  axis=1)
        hr_events_df.columns = ['Starting Timestamp', 'Ending Timestamp']
        hr_events_df = hr_events_df.loc[desired_time_index, :]
        # hr_events_df.to_csv(rootPath+"timestamp_" +dir+ "_EDA.csv")
        
        try:
            for i in range(len(hr_events_df['Starting Timestamp'])):
                # print(hr_events_df['Starting Timestamp'][i+1])
                for j in range(len(hr_events_df['Ending Timestamp'])):
                    if ((hr_events_df['Starting Timestamp'][i+1] - hr_events_df['Ending Timestamp'][i]) < datetime.timedelta(minutes=10) and 
                        (hr_events_df['Starting Timestamp'][i+1] - hr_events_df['Ending Timestamp'][i]) > datetime.timedelta(minutes=0, seconds=0)):
                        hr_events_df['Starting Timestamp'][i+1] = hr_events_df['Ending Timestamp'][i]
                        # print(hr_events_df['Starting Timestamp'][i+1], '\n', hr_events_df['Ending Timestamp'][i])    
                    else:
                        pass     
        except KeyError as error:
            logging.info('range error in delta time adjustments') 
        
        fig, ax4 = plt.subplots(figsize=(30, 10))
        ax4.plot(hr_df['datetime'], hr_df['MAG_K500'], color='red')
        for d in hr_events_df.index:
            ax4.axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp']
                        [d], facecolor="g", edgecolor="none", alpha=0.5)
        ax4.relim()
        ax4.autoscale_view()
        fig.savefig('./Plots/Heart_Rate_figure.png')


def stack_plot_results():
    
    # ema_df = pd.read_csv('L:/C O D E/UMBC VR Project/Stress_Detection/EMA_Survey/ema.csv')
    # ema_df = ema_df.iloc[2: , :]
    # ema_df.reset_index(inplace=True)

    # date_range = []

    # for i in range(len(ema_df)):
    #     date_range.append(('Survey Start Time = ' + ema_df.StartDate[i], 'Survey End Time = ' + ema_df.EndDate[i], 'Break = ' + ema_df.Break[i],
    #                     'Rushed = ' + ema_df.Rushed[i], 'Confront_Authority = '+ ema_df.Confront_authority[i],
    #             'Rude_Family = ' + ema_df.Rude_family[i], 'Gen_Disrespect= ' + ema_df.gen_disrespect[i],
    #             'COVID_concern= ' + ema_df.COVID_concern[i], 'Discomfort= ' + ema_df.Discomfort[i],
    #             'Lack_support= ' + ema_df.Lack_support[i],   'Team_value= ' + ema_df.Team_value[i],
    #             'Demands= ' + ema_df.Demands[i], 'Death= ' + ema_df.Death[i], 
    #             'Other_work-stress= ' + ema_df['Other_work-stress'][i],  'Other_non-work-stress= ' + ema_df['Other_non-work-stress'][i] ))
             
    # Print_start_date_range = str(date_range[0]).replace("'", "")
    # Print_end_date_range = str(date_range[1]).replace("'", "")  

    # fig = plt.figure(figsize=(70,10))
    # fig, ax2 = plt.subplots(figsize=(50,15)) 
    # ax2 = fig.add_subplot(121)
    
            
############################################################   EMA Survey Plot ################################################################
    ema_df = pd.read_csv('./EMA_Survey/ema.csv')
    ema_df = ema_df.iloc[2: , :]
    ema_df.reset_index(inplace=True)
    forenoon_ema_df = ema_df.iloc[[0], :]
    afternoon_ema_df = ema_df.iloc[[1], :]

    forenoon_data = []
    forenoon_data.append('Start Time = ' + str(forenoon_ema_df['StartDate'].values))
    forenoon_data.append('End Time = '+  str(forenoon_ema_df['EndDate'].values))

    if (int(forenoon_ema_df['Break'].values)) > 0 and (int(forenoon_ema_df['Break'].values)) < 8:
        forenoon_data.append('Break = ' + (str(forenoon_ema_df['Break'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Rushed'].values)) > 0 and (int(forenoon_ema_df['Rushed'].values)) < 8:
        forenoon_data.append('Rushed = ' + (str(forenoon_ema_df['Rushed'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Confront_authority'].values)) > 0 and (int(forenoon_ema_df['Confront_authority'].values)) < 8:
        forenoon_data.append('Confront_authority = ' + (str(forenoon_ema_df['Confront_authority'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Rude_family'].values)) > 0 and (int(forenoon_ema_df['Rude_family'].values)) < 8:
        forenoon_data.append('Rude_family = ' + (str(forenoon_ema_df['Rude_family'].values)))
    else:
        pass

    if (int(forenoon_ema_df['gen_disrespect'].values)) > 0 and (int(forenoon_ema_df['gen_disrespect'].values)) < 8:
        forenoon_data.append('gen_disrespect = ' + (str(forenoon_ema_df['gen_disrespect'].values)))
    else:
        pass

    if (int(forenoon_ema_df['COVID_concern'].values)) > 0 and (int(forenoon_ema_df['COVID_concern'].values)) < 8:
        forenoon_data.append('COVID_concern = ' + (str(forenoon_ema_df['COVID_concern'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Discomfort'].values)) > 0 and (int(forenoon_ema_df['Discomfort'].values)) < 8:
        forenoon_data.append('Discomfort = ' + (str(forenoon_ema_df['Discomfort'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Lack_support'].values)) > 0 and (int(forenoon_ema_df['Lack_support'].values)) < 8:
        forenoon_data.append('Lack_support = ' + (str(forenoon_ema_df['Lack_support'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Team_value'].values)) > 0 and (int(forenoon_ema_df['Team_value'].values)) < 8:
        forenoon_data.append('Team_value = ' + (str(forenoon_ema_df['Team_value'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Demands'].values)) > 0 and (int(forenoon_ema_df['Demands'].values)) < 8:
        forenoon_data.append('Demands = ' + (str(forenoon_ema_df['Demands'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Death'].values)) > 0 and (int(forenoon_ema_df['Death'].values)) < 8:
        forenoon_data.append('Death = ' + (str(forenoon_ema_df['Death'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Other_work-stress'].values)) > 0 and (int(forenoon_ema_df['Other_work-stress'].values)) < 8:
        forenoon_data.append('Other_work-stress = ' + (str(forenoon_ema_df['Other_work-stress'].values)))
    else:
        pass

    if (int(forenoon_ema_df['Other_non-work-stress'].values)) > 0 and (int(forenoon_ema_df['Other_non-work-stress'].values)) < 8:
        forenoon_data.append('Other_non-work-stress = ' + (str(forenoon_ema_df['Other_non-work-stress'].values)))
    else:
        pass

    afternoon_data = []
    afternoon_data.append('Start Time = ' + str(afternoon_ema_df['StartDate'].values))
    afternoon_data.append('End Time = '+  str(afternoon_ema_df['EndDate'].values))

    if (int(afternoon_ema_df['Break'].values)) > 0 and (int(afternoon_ema_df['Break'].values)) < 8:
        afternoon_data.append('Break = ' + (str(afternoon_ema_df['Break'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Rushed'].values)) > 0 and (int(afternoon_ema_df['Rushed'].values)) < 8:
        afternoon_data.append('Rushed = ' + (str(afternoon_ema_df['Rushed'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Confront_authority'].values)) > 0 and (int(afternoon_ema_df['Confront_authority'].values)) < 8:
        afternoon_data.append('Confront_authority = ' + (str(afternoon_ema_df['Confront_authority'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Rude_family'].values)) > 0 and (int(afternoon_ema_df['Rude_family'].values)) < 8:
        afternoon_data.append('Rude_family = ' + (str(afternoon_ema_df['Rude_family'].values)))
    else:
        pass

    if (int(afternoon_ema_df['gen_disrespect'].values)) > 0 and (int(afternoon_ema_df['gen_disrespect'].values)) < 8:
        afternoon_data.append('gen_disrespect = ' + (str(afternoon_ema_df['gen_disrespect'].values)))
    else:
        pass

    if (int(afternoon_ema_df['COVID_concern'].values)) > 0 and (int(afternoon_ema_df['COVID_concern'].values)) < 8:
        afternoon_data.append('COVID_concern = ' + (str(afternoon_ema_df['COVID_concern'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Discomfort'].values)) > 0 and (int(afternoon_ema_df['Discomfort'].values)) < 8:
        afternoon_data.append('Discomfort = ' + (str(afternoon_ema_df['Discomfort'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Lack_support'].values)) > 0 and (int(afternoon_ema_df['Lack_support'].values)) < 8:
        afternoon_data.append('Lack_support = ' + (str(afternoon_ema_df['Lack_support'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Team_value'].values)) > 0 and (int(afternoon_ema_df['Team_value'].values)) < 8:
        afternoon_data.append('Team_value = ' + (str(afternoon_ema_df['Team_value'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Demands'].values)) > 0 and (int(afternoon_ema_df['Demands'].values)) < 8:
        afternoon_data.append('Demands = ' + (str(afternoon_ema_df['Demands'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Death'].values)) > 0 and (int(afternoon_ema_df['Death'].values)) < 8:
        afternoon_data.append('Death = ' + (str(afternoon_ema_df['Death'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Other_work-stress'].values)) > 0 and (int(afternoon_ema_df['Other_work-stress'].values)) < 8:
        afternoon_data.append('Other_work-stress = ' + (str(afternoon_ema_df['Other_work-stress'].values)))
    else:
        pass

    if (int(afternoon_ema_df['Other_non-work-stress'].values)) > 0 and (int(afternoon_ema_df['Other_non-work-stress'].values)) < 8:
        afternoon_data.append('Other_non-work-stress = ' + (str(afternoon_ema_df['Other_non-work-stress'].values)))
    else:
        pass

        
    rep_lst = ['[', ']', '"', "'"]
    for i in rep_lst:
        if i in str(afternoon_data):
            afternoon_data = str(afternoon_data).replace(i, '')
            
    rep_lst = ['[', ']', '"', "'"]
    for i in rep_lst:
        if i in str(forenoon_data):
            forenoon_data = str(forenoon_data).replace(i, '')

########################################################################################################################################

    
    
    print('\n', '******************************* Preparing for combined chart ****************************************', '\n')
    fig, axs = plt.subplots(nrows=5, sharex=True, subplot_kw=dict(
        frameon=False), figsize=(30, 15))  # frameon=False removes frames
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()
    axs[4].grid()
    
    plt.figtext(0.1, .95, forenoon_data, ha="left", fontsize=10, bbox={"facecolor":"orange", "alpha":2.0, "pad":10}, wrap= True)
    plt.figtext(0.1, .90, afternoon_data, ha="left", fontsize=10, bbox={"facecolor":"pink", "alpha":2.0, "pad":10},wrap = True)
    

    axs[0].plot(acc_df['datetime'], acc_df['MAG_K500'],
                color='b', label="Acc Magnitude")
    axs[1].plot(hr_df['datetime'], hr_df['MAG_K500'],
                color='m', label="Heart rate")
    axs[2].plot(hrv_features['datetime'], hrv_features['MAG_K500'],
                color='g',  label="HR-Variability")
    axs[3].plot(eda_phasic_df['datetime'], eda_phasic_df['MAG_K500'],
                color='r', label="EDA-Phasic")
    axs[4].plot(eda_tonic_df['datetime'], eda_tonic_df['MAG_K500'],
                color='c', label="EDA-Tonic")

    for d in tqdm(hrv_events_df.index):
        axs[0].axvspan(hrv_events_df['Starting Timestamp'][d],
                       hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[1].axvspan(hrv_events_df['Starting Timestamp'][d],
                       hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[2].axvspan(hrv_events_df['Starting Timestamp'][d],
                       hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[3].axvspan(hrv_events_df['Starting Timestamp'][d],
                       hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[4].axvspan(hrv_events_df['Starting Timestamp'][d],
                       hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)

    for d in tqdm(acc_events_df.index):
        axs[0].axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[1].axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[2].axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[3].axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[4].axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        

    for d in tqdm(eda_phasic_events_df.index):
        axs[0].axvspan(eda_phasic_events_df['Starting Timestamp'][d], eda_phasic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[1].axvspan(eda_phasic_events_df['Starting Timestamp'][d], eda_phasic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[2].axvspan(eda_phasic_events_df['Starting Timestamp'][d], eda_phasic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[3].axvspan(eda_phasic_events_df['Starting Timestamp'][d], eda_phasic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[4].axvspan(eda_phasic_events_df['Starting Timestamp'][d], eda_phasic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)

    for d in tqdm(eda_tonic_events_df.index):
        axs[0].axvspan(eda_tonic_events_df['Starting Timestamp'][d], eda_tonic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[1].axvspan(eda_tonic_events_df['Starting Timestamp'][d], eda_tonic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[2].axvspan(eda_tonic_events_df['Starting Timestamp'][d], eda_tonic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[3].axvspan(eda_tonic_events_df['Starting Timestamp'][d], eda_tonic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[4].axvspan(eda_tonic_events_df['Starting Timestamp'][d], eda_tonic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)

    for d in tqdm(hr_events_df.index):
        axs[0].axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[1].axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[2].axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[3].axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)
        axs[4].axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=1)


#gbrcm 

    for i in range(5):
        axs[i].legend()
        # axs.xaxis.set_major_locator(mdates.MinuteLocator(interval=20))   #to get a tick every 15 minutes
        # axs.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
        # plt.gcf().autofmt_xdate()
        

    import matplotlib.dates as mdates
    axs[4].xaxis.set_major_locator(mdates.MinuteLocator(interval=20))   #to get a tick every 15 minutes
    axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
    plt.gcf().autofmt_xdate()
    
    fig.savefig('./Stacked_charts/Stressful_Regions.png')
    # ("./Metadata/"+ dir+"_HRV.csv")


Extract_HRV_Information()
Extract_ACC_Infromation()
Extract_GSR_Phasic_Information()
Extract_GSR_Tonic_Information()
Extract_Heart_Rate_Features()
stack_plot_results()


# if __name__ == "__main__":
#     # creating processes
#     p1 = multiprocessing.Process(Extract_HRV_Information())
#     p2 = multiprocessing.Process(Extract_ACC_Infromation())
#     p3 = multiprocessing.Process(Extract_GSR_Phasic_Information())
#     p4 = multiprocessing.Process(Extract_GSR_Tonic_Information)
#     p5 = multiprocessing.Process(Extract_Heart_Rate_Features())
#     p6 = multiprocessing.Process(stack_plot_results)
  
#     # starting process 1
#     p1.start()
#     # starting process 2
#     p2.start()
#     p3.start()
#     p4.start()
#     p5.start()
#     p6.start()
  
#     # wait until process 1 is finished
#     p1.join()
#     # wait until process 2 is finished
#     p2.join()
#     p3.join()
#     p4.join()
#     p5.join()
#     p6.join()


  
#     # both processes finished
#     print("Done!")