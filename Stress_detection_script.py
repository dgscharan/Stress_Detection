import plotly
import os
import sys
import time 
import argparse
import numpy as np
import pandas as pd
import zipfile,fnmatch
import flirt.reader.empatica
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxopt as cv
import cvxopt.solvers
from neurokit2 import eda_phasic

# rootPath = r"./"
# pattern = '*.zip'

rootPath = input("Enter Folder Path : ")
pattern = input("Enter File Name : ")


for root, dirs, files in os.walk(rootPath):
    for filename in fnmatch.filter(files, pattern):
        print(os.path.join(root, filename))
        zipfile.ZipFile(os.path.join(root, filename)).extractall(os.path.join(root, os.path.splitext(filename)[0]))

dir = os.path.splitext(pattern)[0]

# os.listdir(dir)

def Extract_HRV_Information():
    global hrv_features
    global hrv_events_df 
    
    ibi = pd.read_csv(rootPath+'/'+dir+'\IBI.csv')
    mean_ibi = ibi[' IBI'].mean()
    average_heart_rate = 60/mean_ibi
    print('mean ibi is :', mean_ibi)
    print('mean heart rate :', average_heart_rate.round())
    ibis = flirt.reader.empatica.read_ibi_file_into_df(rootPath+'/'+dir+ '\IBI.csv')
    hrv_features = flirt.get_hrv_features(ibis['ibi'], 128, 1, ["td", "fd"], 0.2)
    hrv_features = hrv_features.dropna(how='any',axis=0)
    hrv_features.reset_index(inplace=True)
    
    def moving_avarage_smoothing(X,k):
        S = np.zeros(X.shape[0])
        for t in tqdm(range(X.shape[0]), desc="Processing HRV data"):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
        return S

    MAG_K500  = moving_avarage_smoothing(hrv_features['hrv_rmssd'], 500)
    hrv_features['MAG_K500'] = MAG_K500
    
    hrv_features.to_csv("./Metadata/"+ dir+"_HRV.csv")
    # hrv_features.to_csv(os.path.join('./Metadata'+dir+'_HRV.csv'))
    mean_rmssd = hrv_features['hrv_rmssd'].mean()
    std_rmssd = hrv_features['hrv_rmssd'].std()
    
    def deviation_above_mean_hrv(unit):
        if unit == 0:
            return (mean_rmssd)
        else:
            return (mean_rmssd + (unit*std_rmssd))

    def Starting_timeStamp(column, time_frames):
        starting_time_index = []
        for i in range(len(column)-1):
            if column[i] < deviation_above_mean_hrv(1) and column[i+1] > deviation_above_mean_hrv(1):
                starting_time_index.append(time_frames[i])
        return starting_time_index
            
    def Ending_timeStamp(column, time_frames):
        time_index = []
        for i in range(len(column)-1):
            if column[i] > deviation_above_mean_hrv(1) and column[i+1] < deviation_above_mean_hrv(1):
                time_index.append(time_frames[i])
        if column[len(column) -1 ] > deviation_above_mean_hrv(1):
            time_index.insert(len(time_index), time_frames[len(time_frames) -1])
        else:
            pass        
        return time_index
    
    starting_timestamp = Starting_timeStamp(hrv_features['MAG_K500'], hrv_features['datetime'])
    ending_timestamp = Ending_timeStamp(hrv_features['MAG_K500'], hrv_features['datetime'])
    
    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30,10)) 
        ax2.plot(hrv_features['datetime'],hrv_features['MAG_K500'],color='red')
        fig.savefig('./Plots/HRV_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        else:
            pass
        
        difference = [] # initialization of result list
        time_delta_minutes = []
        desired_time_index = []

        zip_object = zip(ending_timestamp, starting_timestamp)

        for list1_i, list2_i in zip_object:
            difference.append(list1_i-list2_i)# append each difference to list
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
        
        fig, ax = plt.subplots(figsize=(20,6)) 
        ax.plot(hrv_features['datetime'],hrv_features['MAG_K500'],color='red')

        for d in hrv_events_df.index:
            # print(events_df['Starting Timestamp'], events_df['Ending Timestamp'])
            ax.axvspan(hrv_events_df['Starting Timestamp'][d], hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        ax.relim()
        ax.autoscale_view()
        fig.savefig('./Plots/HRV_figure.png')
        return hrv_features,hrv_events_df 
        
# Extract_HRV_Information()

def Extract_ACC_Infromation():
    global acc_df
    global acc_events_df
    
    acc_df = pd.read_csv(rootPath+'/'+dir + '/ACC.csv')
    acc_df = flirt.reader.empatica.read_acc_file_into_df(rootPath+'/'+dir + '/ACC.csv')
    acc_df['Magnitude'] = np.sqrt(acc_df['acc_x']**2 + acc_df['acc_y']**2 + acc_df['acc_z']**2)
    print("Magnitude Mean : ", acc_df['Magnitude'].mean())
    acc_df.reset_index(inplace=True)
    
    print("--------------Smoothing The ACC Curve----------------")
    def moving_avarage_smoothing(X,k):
        S = np.zeros(X.shape[0])
        for t in tqdm(range(X.shape[0]), desc ="Processing ACC data"):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
        return S

    MAG_K500  = moving_avarage_smoothing(acc_df['Magnitude'], 15000)
    acc_df['MAG_K500'] = MAG_K500
    
    acc_df.to_csv("./Metadata/"+ dir+"_ACC.csv")

    mean_acc_magnitude = acc_df['Magnitude'].mean()
    std_acc_magnitude = acc_df['Magnitude'].std()
    print("Average Magnitude of the Acc Data : ", mean_acc_magnitude)
    one_std_above_mean_acc = mean_acc_magnitude + std_acc_magnitude
    
    def deviation_above_mean_acc(unit):
        if unit == 0:
            return (mean_acc_magnitude)
        else:
            return (mean_acc_magnitude + (unit*std_acc_magnitude))
    print('----------Checking if possible regions exist-----------')
    def Starting_timeStamp(column, time_frames):
        starting_time_index = []
        for i in range(len(column)-1):
            if column[i] < deviation_above_mean_acc(1) and column[i+1] > deviation_above_mean_acc(1):
                starting_time_index.append(time_frames[i])
        return starting_time_index
        
    def Ending_timeStamp(column, time_frames):
        time_index = []
        for i in range(len(column)-1):
            if column[i] > deviation_above_mean_acc(1) and column[i+1] < deviation_above_mean_acc(1):
                time_index.append(time_frames[i])
        if column[len(column) -1 ] > deviation_above_mean_acc(1):
            time_index.insert(len(time_index), time_frames[len(time_frames) -1])
        else:
            pass        
        return time_index

    starting_timestamp = Starting_timeStamp(acc_df['MAG_K500'], acc_df['datetime'])
    ending_timestamp = Ending_timeStamp(acc_df['MAG_K500'], acc_df['datetime'])

    print('------------- Processing the Plot -------------')
    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30,10)) 
        ax2.plot(acc_df['datetime'],acc_df['MAG_K500'],color='red')
        fig.savefig('./Plots/ACC_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        difference = [] # initialization of result list
        time_delta_minutes = []
        desired_time_index = []
        zip_object = zip(ending_timestamp, starting_timestamp)

        for list1_i, list2_i in zip_object:
            difference.append(list1_i-list2_i)# append each difference to list
            
        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)
            
        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 3.00:
                desired_time_index.append(i)
                
        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        acc_events_df = pd.concat(frames,  axis=1)
        acc_events_df.columns = ['Starting Timestamp', 'Ending Timestamp']
        acc_events_df = acc_events_df.loc[desired_time_index, :]
        # acc_events_df.to_csv(rootPath+"timestamp_" +dir+ "_ACC.csv")

        fig, ax2 = plt.subplots(figsize=(30,10)) 
        ax2.plot(acc_df['datetime'],acc_df['MAG_K500'],color='red')

        for d in acc_events_df.index:
            # print(events_df['Starting Timestamp'], events_df['Ending Timestamp'])
            ax2.axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        ax2.relim()
        ax2.autoscale_view()
        fig.savefig('./Plots/ACC_figure.png')

# Extract_ACC_Infromation()

def Extract_GSR_Phasic_Information():
    global eda_df
    global eda_phasic_df
    global eda_phasic_events_df
    
    eda_df = pd.read_csv(rootPath+'/'+dir+'/EDA.csv')
    eda_df = flirt.reader.empatica.read_eda_file_into_df(rootPath+'/'+ dir + '/EDA.csv')
    eda_df.reset_index(inplace=True)
    eda = np.array(eda_df['eda'])
    Phasic_Tonic_DF = eda_phasic(eda,4, method = 'cvxEDA')
    eda_df['tonic'] = Phasic_Tonic_DF['EDA_Tonic']
    eda_df['phasic'] = Phasic_Tonic_DF['EDA_Phasic']
    eda_phasic_df = eda_df.copy()

    def moving_avarage_smoothing(X,k):
        S = np.zeros(X.shape[0])
        for t in tqdm(range(X.shape[0]), desc ="Processing GSR Phasic data"):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
        return S

    MAG_K500  = moving_avarage_smoothing(eda_phasic_df['phasic'], 2000)
    eda_phasic_df['MAG_K500'] = MAG_K500
    # hrv_features.to_csv('hrv_features.csv')
    mean_eda_phasic = eda_phasic_df['phasic'].mean()
    std_eda_phasic = eda_phasic_df['phasic'].std()

    def deviation_above_mean_eda_phasic(unit):
        if unit == 0:
            return (mean_eda_phasic)
        else:
            return (mean_eda_phasic + (unit*std_eda_phasic))

    def Starting_timeStamp(column, time_frames):
        starting_time_index = []
        for i in range(len(column)-1):
            if column[i] < deviation_above_mean_eda_phasic(1) and column[i+1] > deviation_above_mean_eda_phasic(1):
                starting_time_index.append(time_frames[i])
        return starting_time_index
            
    def Ending_timeStamp(column, time_frames):
        time_index = []
        for i in range(len(column)-1):
            if column[i] > deviation_above_mean_eda_phasic(1) and column[i+1] < deviation_above_mean_eda_phasic(1):
                time_index.append(time_frames[i])
        if column[len(column) -1 ] > deviation_above_mean_eda_phasic(1):
            time_index.insert(len(time_index), time_frames[len(time_frames) -1])
        else:
            pass        
        return time_index
    
    starting_timestamp = Starting_timeStamp(eda_phasic_df['MAG_K500'], eda_phasic_df['datetime'])
    ending_timestamp = Ending_timeStamp(eda_phasic_df['MAG_K500'], eda_phasic_df['datetime'])
    print('------------- Processing the Plot -------------')
    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30,10)) 
        ax2.plot(eda_phasic_df['datetime'],eda_phasic_df['MAG_K500'],color='red')
        fig.savefig('./Plots/EDA_Phasic_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        
        difference = [] # initialization of result list
        time_delta_minutes = []
        desired_time_index = []

        zip_object = zip(ending_timestamp, starting_timestamp)

        for list1_i, list2_i in zip_object:
            difference.append(list1_i-list2_i)# append each difference to list
            
        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)
            
        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 2.00:
                desired_time_index.append(i) 
                
        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        eda_phasic_events_df = pd.concat(frames,  axis=1)
        eda_phasic_events_df.columns = ['Starting Timestamp', 'Ending Timestamp']
        eda_phasic_events_df = eda_phasic_events_df.loc[desired_time_index, :]
        eda_phasic_events_df.to_csv(rootPath+"timestamp_" +dir+ "_EDA.csv")

        fig, ax3 = plt.subplots(figsize=(30,10)) 
        ax3.plot(eda_phasic_df['datetime'],eda_phasic_df['MAG_K500'],color='red')

        for d in eda_phasic_events_df.index:
            # print(events_df['Starting Timestamp'], events_df['Ending Timestamp'])
            ax3.axvspan(eda_phasic_events_df['Starting Timestamp'][d], eda_phasic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        ax3.relim()
        ax3.autoscale_view()
        fig.savefig('./Plots/EDA_Phasic_figure.png')
        return eda_df

# Extract_GSR_Phasic_Information()

def Extract_GSR_Tonic_Information():
    global eda_tonic_df
    global eda_tonic_events_df
    
    eda_df = pd.read_csv(rootPath+'/'+dir+'/EDA.csv')
    eda_df = flirt.reader.empatica.read_eda_file_into_df( rootPath+'/'+ dir + '/EDA.csv')
    eda_df.reset_index(inplace=True)
    eda = np.array(eda_df['eda'])
    Phasic_Tonic_DF = eda_phasic(eda,4, method = 'cvxEDA')
    eda_df['tonic'] = Phasic_Tonic_DF['EDA_Tonic']
    eda_df['phasic'] = Phasic_Tonic_DF['EDA_Phasic']
    eda_tonic_df = eda_df.copy()
    
    def moving_avarage_smoothing(X,k):
        S = np.zeros(X.shape[0])
        for t in tqdm(range(X.shape[0]), desc="Processing GSR Tonic Data"):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
        return S
        
    MAG_K500  = moving_avarage_smoothing(eda_tonic_df['tonic'], 2000)
    eda_tonic_df['MAG_K500'] = MAG_K500
    # hrv_features.to_csv('hrv_features.csv')
    mean_eda_tonic = eda_tonic_df['tonic'].mean()
    std_eda_tonic = eda_tonic_df['tonic'].std()
    
    # print(mean_eda_tonic, std_eda_tonic)
    def deviation_above_mean_eda_tonic(unit): 
        if unit == 0:
            return (mean_eda_tonic)
        else:
            return (mean_eda_tonic + (unit*std_eda_tonic))

    def Starting_timeStamp(column, time_frames):
        starting_time_index = []
        for i in range(len(column)-1):
            if column[i] < deviation_above_mean_eda_tonic(1) and column[i+1] > deviation_above_mean_eda_tonic(1):
                starting_time_index.append(time_frames[i])
        return starting_time_index
            
    def Ending_timeStamp(column, time_frames):
        time_index = []
        for i in range(len(column)-1):
            if column[i] > deviation_above_mean_eda_tonic(1) and column[i+1] < deviation_above_mean_eda_tonic(1):
                time_index.append(time_frames[i])
        if column[len(column) -1 ] > deviation_above_mean_eda_tonic(1):
            time_index.insert(len(time_index), time_frames[len(time_frames) -1])
        else:
            pass        
        return time_index
    
    print("extracting starting and ending timestamps")
    starting_timestamp = Starting_timeStamp(eda_tonic_df['MAG_K500'], eda_tonic_df['datetime'])
    ending_timestamp = Ending_timeStamp(eda_tonic_df['MAG_K500'], eda_tonic_df['datetime'])
    
    print("checking time length condition")
    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30,10)) 
        ax2.plot(eda_tonic_df['datetime'],eda_tonic_df['MAG_K500'],color='red')
        fig.savefig('./Plots/EDA_Phasic_figure.png')
    else:
        print("entering final else block")
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
            
        difference = [] # initialization of result list
        time_delta_minutes = []
        desired_time_index = []

        zip_object = zip(ending_timestamp, starting_timestamp)

        for list1_i, list2_i in zip_object:
            difference.append(list1_i-list2_i)# append each difference to list
            
        for i in difference:
            time_delta_minutes.append(i.total_seconds()/60)
            
        for i in range(len(time_delta_minutes)):
            if time_delta_minutes[i] > 2.00:
                desired_time_index.append(i)
        
        starting_timestamp_df = pd.DataFrame(starting_timestamp)
        ending_timestamp_df = pd.DataFrame(ending_timestamp)
        frames = (starting_timestamp_df, ending_timestamp_df)
        eda_tonic_events_df = pd.concat(frames,  axis=1)
        eda_tonic_events_df.columns = ['Starting Timestamp', 'Ending Timestamp']
        eda_tonic_events_df = eda_tonic_events_df.loc[desired_time_index, :]
        # eda_tonic_events_df.to_csv(rootPath+"timestamp_" +dir+ "_EDA.csv")
        fig, ax4 = plt.subplots(figsize=(30,10)) 
        ax4.plot(eda_tonic_df['datetime'],eda_tonic_df['MAG_K500'],color='red')

        for d in eda_tonic_events_df.index:
            # print(events_df['Starting Timestamp'], events_df['Ending Timestamp'])
            ax4.axvspan(eda_tonic_events_df['Starting Timestamp'][d], eda_tonic_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        ax4.relim()
        ax4.autoscale_view()
        fig.savefig('./Plots/EDA_tonic_figure.png')
        
        
# Extract_GSR_Tonic_Information()

def Extract_Heart_Rate_Features():
    global hr_df
    global hr_events_df
    
    hr_df = flirt.reader.empatica.read_hr_file_into_df(rootPath+'/'+dir+'/HR.csv')
    hr_df.reset_index(inplace = True)
    
    def moving_avarage_smoothing(X,k):
        S = np.zeros(X.shape[0])
        for t in tqdm(range(X.shape[0]), desc='Processing Heart Rate data'):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
        return S
    
    MAG_K500  = moving_avarage_smoothing(hr_df['hr'], 500)
    hr_df['MAG_K500'] = MAG_K500
    # hrv_features.to_csv('hrv_features.csv')
    hr_avg = hr_df['MAG_K500'].mean()
    hr_std = hr_df['MAG_K500'].std()
    
    def deviation_above_mean_hr(unit):
        if unit == 0:
            return (hr_avg)
        else:
            return (hr_avg + (unit*hr_std))

    def Starting_timeStamp(column, time_frames):
        starting_time_index = []
        for i in range(len(column)-1):
            if column[i] < deviation_above_mean_hr(1) and column[i+1] > deviation_above_mean_hr(1):
                starting_time_index.append(time_frames[i])
        return starting_time_index
            
    def Ending_timeStamp(column, time_frames):
        time_index = []
        for i in range(len(column)-1):
            if column[i] > deviation_above_mean_hr(1) and column[i+1] < deviation_above_mean_hr(1):
                time_index.append(time_frames[i])
        if column[len(column) -1 ] > deviation_above_mean_hr(1):
            time_index.insert(len(time_index), time_frames[len(time_frames) -1])
        else:
            pass        
        return time_index
    
    starting_timestamp = Starting_timeStamp(hr_df['MAG_K500'], hr_df['datetime'])
    ending_timestamp = Ending_timeStamp(hr_df['MAG_K500'], hr_df['datetime'])

    if len(starting_timestamp) < 1:
        fig, ax2 = plt.subplots(figsize=(30,10)) 
        ax2.plot(hr_df['datetime'],hr_df['MAG_K500'],color='red')
        fig.savefig('./Plots/Heart_rate_figure.png')
    else:
        if starting_timestamp > ending_timestamp:
            ending_timestamp.pop(0)
        difference = [] # initialization of result list
        time_delta_minutes = []
        desired_time_index = []

        zip_object = zip(ending_timestamp, starting_timestamp)

        for list1_i, list2_i in zip_object:
            difference.append(list1_i-list2_i)# append each difference to list
            
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

        fig, ax4 = plt.subplots(figsize=(30,10)) 
        ax4.plot(hr_df['datetime'],hr_df['MAG_K500'],color='red')

        for d in hr_events_df.index:
            # print(events_df['Starting Timestamp'], events_df['Ending Timestamp'])
            ax4.axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        ax4.relim()
        ax4.autoscale_view()
        fig.savefig('./Plots/Heart_Rate_figure.png')
        
def stack_plot_results():
    print('\n', '******************************* Preparing for combined chart ****************************************', '\n')
    fig, axs = plt.subplots(nrows=5, sharex=True, subplot_kw=dict(frameon=False),figsize = (30,15) ) # frameon=False removes frames
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()
    axs[4].grid()

    axs[0].plot(acc_df['datetime'],acc_df['MAG_K500'],color='red', label="Acc Magnitude")
    axs[1].plot(hr_df['datetime'], hr_df['MAG_K500'],color='red', label="Heart rate")
    axs[2].plot(hrv_features['datetime'],hrv_features['MAG_K500'],color='red',  label="HR-Variability")
    axs[3].plot(eda_phasic_df['datetime'],eda_phasic_df['MAG_K500'],color='red', label="EDA-Phasic")
    axs[4].plot(eda_tonic_df['datetime'], eda_tonic_df['MAG_K500'], color='red', label="EDA-Tonic")
    

    for d in tqdm(hrv_events_df.index):
        axs[0].axvspan(hrv_events_df['Starting Timestamp'][d], hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        axs[1].axvspan(hrv_events_df['Starting Timestamp'][d], hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        axs[2].axvspan(hrv_events_df['Starting Timestamp'][d], hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        axs[3].axvspan(hrv_events_df['Starting Timestamp'][d], hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)
        axs[4].axvspan(hrv_events_df['Starting Timestamp'][d], hrv_events_df['Ending Timestamp'][d], facecolor="g", edgecolor="none", alpha=0.5)

        

    # for d in acc_events_df.index:
    #     axs[1].axvspan(acc_events_df['Starting Timestamp'][d], acc_events_df['Ending Timestamp'][d], facecolor="b", edgecolor="none", alpha=0.5)

    # for d in eda_phasic_events_df.index:
    #     axs[2].axvspan(eda_phasic_events_df['Starting Timestamp'][d], eda_phasic_events_df['Ending Timestamp'][d], facecolor="b", edgecolor="none", alpha=0.5)

    # for d in eda_tonic_events_df.index:
    #     axs[3].axvspan(eda_tonic_events_df['Starting Timestamp'][d], eda_tonic_events_df['Ending Timestamp'][d], facecolor="b", edgecolor="none", alpha=0.5)

    # for d in hr_events_df.index:
    #     axs[4].axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="b", edgecolor="none", alpha=0.5) 

    for i in range(5):
        axs[i].legend()
    fig.savefig('./Stacked_charts/Stressful_Regions.png')
    # ("./Metadata/"+ dir+"_HRV.csv")
        
Extract_HRV_Information()
Extract_ACC_Infromation()
Extract_GSR_Phasic_Information()
Extract_GSR_Tonic_Information()
Extract_Heart_Rate_Features()
stack_plot_results()
