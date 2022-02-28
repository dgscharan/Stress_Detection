### dummy plot for Dr. Yoon's RA

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import zipfile
import fnmatch
import flirt.reader.empatica
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxopt as cv
import cvxopt.solvers
from neurokit2 import eda_phasic


rootPath = input("Enter Folder Path : ")
pattern = input("Enter File Name : ")
for root, dirs, files in os.walk(rootPath):
    for filename in fnmatch.filter(files, pattern):
        print(os.path.join(root, filename))
        zipfile.ZipFile(os.path.join(root, filename)).extractall(
            os.path.join(root, os.path.splitext(filename)[0]))
dir = os.path.splitext(pattern)[0]


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
        fig, ax4 = plt.subplots(figsize=(30, 10))
        ax4.plot(hr_df['datetime'], hr_df['MAG_K500'], color='red')
        for d in hr_events_df.index:
            ax4.axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp']
                        [d], facecolor="g", edgecolor="none", alpha=0.5)
        ax4.relim()
        ax4.autoscale_view()
        
        
        # plt.figtext(0.1, .20, frames, ha='left', fontsize=10, bbox={"facecolor":"pink", "alpha":2.0, "pad":10},wrap = True)
    
        ax5 = fig.add_subplot(122)
        font_size=14
        bbox=[0, 0, 1, 1]
        ax5.axis('off')
        mpl_table = ax5.table(cellText = hr_events_df.values, rowLabels = hr_events_df.index, bbox=bbox, colLabels = hr_events_df.columns)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        fig.savefig('./Plots/Heart_Rate_figure.png')
        
        
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


########################################################################################################################################

    


def stack_plot_results():
    
    plt.rcParams["figure.autolayout"] = True
    print('\n', '******************************* Preparing for combined chart ****************************************', '\n')
    fig, ax1 = plt.subplots(nrows=1, sharex=True, subplot_kw=dict(
        frameon=False), figsize=(30, 15))  # frameon=False removes frames
    ax1.grid()
    # axs[1].grid()
    # axs[2].grid()
    # axs[3].grid()
    # axs[4].grid()
    
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
    
    plt.figtext(0.1, .95, forenoon_data, ha="left", fontsize=10, bbox={"facecolor":"orange", "alpha":2.0, "pad":10}, wrap= True)
    plt.figtext(0.1, .90, afternoon_data, ha="left", fontsize=10, bbox={"facecolor":"pink", "alpha":2.0, "pad":10},wrap = True)
    # plt.figtext(0.1, 0.1, hr_events_df.values.all() , ha="left", fontsize=10, bbox={"facecolor":"pink", "alpha":2.0, "pad":10},wrap = True)


    # ax1.plot(hr_df['datetime'], hr_df['MAG_K500'],
    #             color='red', label="Heart rate")

    for d in hr_events_df.index:
        ax1.axvspan(hr_events_df['Starting Timestamp'][d], hr_events_df['Ending Timestamp'][d], facecolor="b", edgecolor="none", alpha=0.5)
        
    
        
    ax2 = fig.add_subplot(122)
    font_size = 14
    bbox = [0, 0, 1, 1]
    ax2.axis('off')
    mpl_table = ax2.table(cellText=hr_events_df.values , rowLabels= hr_events_df.index, bbox=bbox, colLabels= hr_events_df.columns)
    # plt.show()
    
    # ax1 = fig.add_subplot(122)
    # font_size=14
    # # bbox=[1, 0, 1, 1]
    # ax1.axis('off')
    # mpl_table = ax1.table(cellText = hr_events_df.values, bbox = bbox, rowLabels = hr_events_df.index, colLabels = hr_events_df.columns)
    # mpl_table.auto_set_font_size(False)
    # mpl_table.set_fontsize(font_size)

    # for i in range(5):
    #     axs[i].legend()
    fig.savefig('./Stacked_charts/Stressful_Regions.png')
    # ("./Metadata/"+ dir+"_HRV.csv")

Extract_Heart_Rate_Features()
stack_plot_results()