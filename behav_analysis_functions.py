# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:11:18 2022

@author: kdani
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import ephys

#%%

def load_files(path, extension):
    ###
    # Auxiliary function for loading all file names with certain extention from a folder
    # into a list.
    #
    ###
    paths = glob.glob(path + "\*." + extension)
    
    return paths

def read_bvs(filepath, framerate):
    
    ### 
    # Function to read bsv output sequence of BehaView
    #
    # filepath - string, path to the file
    # framerate - framerate of the video for proper transformation of time
    #             to video frame number
    #
    # Returns data frame with two columns: frame and behavior
    ###
    
    df = pd.read_csv(filepath)
    behav = df["[BehaViewSeq]"].str.split(" - ", expand=True)
    df = behav[0].str.split(":", expand=True)
    df = df.merge(behav[1], left_index=True, right_index=True)
    
    df = df.iloc[:,0:4].astype('int32')
    
    df.iloc[:, 0] *= 3600000
    df.iloc[:, 1] *= 60000
    df.iloc[:, 2] *= 1000
    df.iloc[:, 3] *= 10
    
    milis = df.iloc[:, 0:4].sum(axis=1)
    framerate = 1000/framerate
    milis = milis/framerate
    milis = milis.astype(int)
    
    df = pd.DataFrame(milis).join(behav[1])
    df.columns = ['frame', 'behavior']
    
    return df

def starts_stops(behavior):
    ###
    # Finds when instances of behaviors start and stop 
    #
    # !!!Auxiliary function, don't change!!!
    ###
    behav = pd.DataFrame({"start":behavior["frame"].iloc[::2].values, 
                          'stop':behavior["frame"].iloc[1::2].values})
    behav = behav.astype({"start": int, "stop": int})
    name = pd.unique(behavior.iloc[:,1])
    behav["behavior"] = name[0]
    behav["index"] = 0
        
    return behav

def conditions_ready(file, df, TS, framerate, video_len, test_dir):
    ###
    # Applies conditions and extracts instances when conditions are met 
    # file - path of analyzed file used for filename extraction
    # df - output of read_bvs function
    # TS - filepath with timestamps of frames relative to ephys recordings
    # framerate - framerate of the video
    # video_len - length of the video in frames
    #
    # Saves files into a specified location - temporarily hardcoded in the function
    ###    

    TS = pd.read_csv(TS, header=0, index_col=None)
    behaviors = list(pd.unique(df.iloc[:,1]))

    dataframes = []

    for behavior in behaviors:
        behav = df.where(df["behavior"] == behavior)
        behav.dropna(inplace=True)
        behav.reset_index(drop=True, inplace=True)
        dataframes.append(behav)

    points = list(map(starts_stops, dataframes))

    behs = []
    for point in points:
        point = point.drop(["index", "stop"], axis=1)
        behs.append(point)

    behaviors = ["rearing", "grooming", "freezing"]

    non_ephys_starts = pd.DataFrame(index = range(video_len), 
                        columns= [behavior + '_non_ephys'.format(behavior) for behavior in behaviors])
    non_ephys_filled = pd.DataFrame(index = range(video_len), 
                        columns= [behavior + '_non_ephys'.format(behavior) for behavior in behaviors])

    ephys_starts = pd.DataFrame(index = range(video_len), 
                    columns= [behavior + '_ephys'.format(behavior) for behavior in behaviors])
    ephys_filled = pd.DataFrame(index = range(video_len), 
                    columns= [behavior + '_ephys'.format(behavior) for behavior in behaviors])


    for behavior in behaviors:
        for beh in behs:
            if beh["behavior"][0] == behavior + "_non_ephys":
                non_ephys_starts[behavior + "_non_ephys"].iloc[beh["start"]] = 1

            elif beh["behavior"][0] == behavior + "_ephys":
                ephys_starts[behavior + "_ephys"].iloc[beh["start"]] = 1

    for behavior in behaviors:
        for point in points:
            if point["behavior"][0] == behavior + "_ephys":
                for i in range(len(point)):
                    name = pd.unique(point.iloc[:,2])
                    ephys_filled[name[0]][point["start"][i] - framerate:point["stop"][i]] = 1
            if point["behavior"][0] == behavior + "_non_ephys":
                for i in range(len(point)):
                    name = pd.unique(point.iloc[:,2])
                    non_ephys_filled[name[0]][point["start"][i] - framerate:point["stop"][i]] = 1

    non_ephys_starts.index = TS["TS"]
    non_ephys_filled.index = TS["TS"]         
    ephys_starts.index = TS["TS"]
    ephys_filled.index = TS["TS"]
    non_ephys_starts = non_ephys_starts.fillna(0)
    non_ephys_filled = non_ephys_filled.fillna(0)
    ephys_starts = ephys_starts.fillna(0)
    ephys_filled = ephys_filled.fillna(0)
    
    filename = Path(file).stem


    for behavior in behaviors:

        both_n_ephys_alligned = np.where((non_ephys_starts[behavior+"_non_ephys"] == 1) & (ephys_filled[behavior+"_ephys"] == 1))
        both_n_ephys_alligned = non_ephys_starts.iloc[both_n_ephys_alligned[0]].index
        np.savetxt(test_dir + "Conditions\\" + filename + "_" + behavior +  "_both_n_ephys_alligned.txt", both_n_ephys_alligned)
        both_ephys_alligned = np.where((non_ephys_filled[behavior+"_non_ephys"] == 1) & (ephys_starts[behavior+"_ephys"] == 1))
        both_ephys_alligned = ephys_starts.iloc[both_ephys_alligned[0]].index
        np.savetxt(test_dir + "Conditions\\" + filename + "_" + behavior + "_both_ephys_alligned.txt", both_ephys_alligned)
        
        non_ephys_only = np.where((non_ephys_starts[behavior+"_non_ephys"] == 1) & (ephys_filled[behavior+"_ephys"] == 0))
        non_ephys_only = non_ephys_starts.iloc[non_ephys_only[0]].index
        np.savetxt(test_dir + "Conditions\\" + filename + "_" + behavior + "_non_ephys_only.txt", non_ephys_only)
        
        ephys_only = np.where((non_ephys_filled[behavior+"_non_ephys"] == 0) & (ephys_starts[behavior+"_ephys"] == 1))
        ephys_only = ephys_starts.iloc[ephys_only[0]].index
        np.savetxt(test_dir + "Conditions\\" + filename + "_" + behavior + "_ephys_only.txt", ephys_only)

def behavior_sequences(file, df, TS, framerate, video_len):
    TS = pd.read_csv(TS, header=0, index_col=None)
    behaviors = list(pd.unique(df.iloc[:,1]))

    dataframes = []

    for behavior in behaviors:
        behav = df.where(df["behavior"] == behavior)
        behav.dropna(inplace=True)
        behav.reset_index(drop=True, inplace=True)
        dataframes.append(behav)

    points = list(map(starts_stops, dataframes))

    behs = []
    for point in points:
        point = point.drop(["index", "stop"], axis=1)
        behs.append(point)

    behaviors = ["rearing", "grooming", "freezing"]

    non_ephys_starts = pd.DataFrame(index = range(video_len), 
                        columns= [behavior + '_non_ephys'.format(behavior) for behavior in behaviors])
    non_ephys_filled = pd.DataFrame(index = range(video_len), 
                        columns= [behavior + '_non_ephys'.format(behavior) for behavior in behaviors])

    ephys_starts = pd.DataFrame(index = range(video_len), 
                    columns= [behavior + '_ephys'.format(behavior) for behavior in behaviors])
    ephys_filled = pd.DataFrame(index = range(video_len), 
                    columns= [behavior + '_ephys'.format(behavior) for behavior in behaviors])


    for behavior in behaviors:
        for beh in behs:
            if beh["behavior"][0] == behavior + "_non_ephys":
                non_ephys_starts[behavior + "_non_ephys"].iloc[beh["start"]] = 1

            elif beh["behavior"][0] == behavior + "_ephys":
                ephys_starts[behavior + "_ephys"].iloc[beh["start"]] = 1

    for behavior in behaviors:
        for point in points:
            if point["behavior"][0] == behavior + "_ephys":
                for i in range(len(point)):
                    name = pd.unique(point.iloc[:,2])
                    ephys_filled[name[0]][point["start"][i] - framerate:point["stop"][i]] = 1
            if point["behavior"][0] == behavior + "_non_ephys":
                for i in range(len(point)):
                    name = pd.unique(point.iloc[:,2])
                    non_ephys_filled[name[0]][point["start"][i] - framerate:point["stop"][i]] = 1

    non_ephys_starts.index = TS["TS"]
    non_ephys_filled.index = TS["TS"]         
    ephys_starts.index = TS["TS"]
    ephys_filled.index = TS["TS"]
    non_ephys_starts = non_ephys_starts.fillna(0)
    non_ephys_filled = non_ephys_filled.fillna(0)
    ephys_starts = ephys_starts.fillna(0)
    ephys_filled = ephys_filled.fillna(0)
    ephys_seq_lengths = pd.DataFrame()
    non_ephys_seq_lengths = pd.DataFrame()
    for behavior in behaviors:
        idx_pairs_ephys = np.where(np.diff(np.hstack(([False],ephys_filled[behavior + "_ephys"]==1,[False]))))[0].reshape(-1,2)
        idx_pairs_non_ephys = np.where(np.diff(np.hstack(([False],non_ephys_filled[behavior + "_non_ephys"]==1,[False]))))[0].reshape(-1,2)
        lengths_ephys = []
        lengths_non_ephys = []
        for pair_e, pair_ne in zip(idx_pairs_ephys, idx_pairs_non_ephys):
            length_e = pair_e[1] - pair_e[0]
            length_ne = pair_ne[1] - pair_ne[0]
            lengths_ephys.append(length_e)
            lengths_non_ephys.append(length_ne)
        ephys_seq_lengths[behavior+"_ephys"] = pd.Series(lengths_ephys)
        non_ephys_seq_lengths[behavior+"_non_ephys"]  = pd.Series(lengths_non_ephys)
        
    filename = Path(file).stem
    ephys_seq_lengths.to_csv(r"C:\Ephys_analysis\Sequences_behavior\\" + filename + "ephys_behav_sequence.csv")
    non_ephys_seq_lengths.to_csv(r"C:\Ephys_analysis\Sequences_behavior\\" + filename + "non_ephys_behav_sequence.csv")

def simba_ready(df, video_len):
    ###
    # Prepares data to format accepted in simba "targets_inserted" file
    #
    # df - DataFrame, output of read_bvs function
    # video_len - int, length of video in number of frames
    #
    # Returns data ready to be saved into a csv 
    ###
    behaviors = list(pd.unique(df.iloc[:,1]))
    
    dataframes = []

    for behavior in behaviors:
        behav = df.where(df["behavior"] == behavior)
        behav.dropna(inplace=True)
        behav.reset_index(drop=True, inplace=True)
        dataframes.append(behav)
        
    points = list(map(starts_stops, dataframes))
        
    data = pd.DataFrame(index=range(video_len),columns=behaviors)
    for point in points:
        for i in range(len(point)):
            name = pd.unique(point.iloc[:,2])
            data[name[0]][point["start"][i]:point["stop"][i]] = 1
        
    return data

def save_data(filename, data):
    ###
    # Auxiliary function for data saving
    # filename - str, uses name of the input file
    # data - DataFrame, output of the simba_ready function
    #
    # Saves data into the same folder as original data as .csv file
    ###
    path = os.path.split(filename)
    filename = os.path.splitext(path[1])
    name = path[0] + "\\" + filename[0] + "_filled.csv"
    
    return data.to_csv(name)    

def save_labels(filename, data, col_name): #!!!needs work, add creating Conditions folder in Ephys NP X folders and save into those!!!
    ###
    # Auxiliary function for label saving
    # filename - str, uses name of the input file
    # data - DataFrame, output of the simba_ready function
    # Saves data into the same folder as original data as .csv file
    ###

    path = os.path.split(filename)
    filename = os.path.splitext(path[1])
    name = path[0] + "\\" + filename[0] + "_" + col_name + "_label.txt"
    
    return data.to_csv(name, index=False, header=False)

def make_behav_dirs(behaviors, save_dir):
    for behavior in behaviors:
        behav_path = save_dir + "\\" + behavior
        if not os.path.exists(behav_path):
            os.makedirs(behav_path)

def list_exp_dirs(tests_dir):
    # Function used to list where ephys experiment trials are stored
    # tes - directory with ephys data subdirectories corresponding to experimental pairs
    # 
    tes = os.listdir(tests_dir)
    tests = []
    for i in tes:
        path = tests_dir + "\\" + i + "\\"
        tests.append(path)
    
    return tests

def exp_dates(tests):
    # Get dates and timestamps of experimental trials for future sorting and filtering
    # tests - list, output of list_exp_dirs function
    date_list = []
    for test in tests:
        var = os.listdir(test + "Conditions\\")
        dates = []
        for v in var[0:1]:
            a = v.split("_")[1]
            if len(dates) < 2:
                dates.append(a)
        date_list.append(dates[0])
    return date_list

def plot_conditons(conditions, behaviors, tests, date_list): #!!!add creation of commined_plots_ephys dir!!! and rewrite to accomodate looping through conditons and plots without explicitly wrriting them
    # Function plotting rasters and psths in 2x4 figures based on required conditions (works per behavior)
    # 
    # conditions - list of str naming conditions from conditions_ready output (saved files)
    # behaviors - list of str naming behaviors used in the experiment
    # tests - list of str, output of list_exp_dirs
    # date_list - list of str, exp_dates    
    
    plt.ioff()
    for test, date in zip(tests, date_list):
        spks_ts, units_id = ephys.read_spikes(test, sampling_rate = 30000.0, read_only = "good")
        for behavior in behaviors:
            behav_path = test + "\\combined_plots_ephys\\" + behavior
            if not os.path.exists(behav_path):
                os.makedirs(behav_path)

            centered = []
            names = []
            sems = []
            al = []
            
            for condition in conditions:
                events_ts = np.loadtxt(test + "Conditions\\" + "camA_" + date + "_compr_" + behavior + "_" + condition + ".txt")                        
                centered_ts = ephys.calc_rasters(spks_ts, events_ts, pre_event = 5.0, post_event = 5.0)
                all_fr, mean_fr, sem_fr, t_vec = ephys.fr_events_binless(centered_ts, 0.2, trunc_gauss = 4,
                                                                    sampling_rate = 30000.0, sampling_out = 100,
                                                                    pre_event = 5.0, post_event = 5.0)
                centered.append(centered_ts)
                names.append(condition)
                sems.append(sem_fr)
                al.append(all_fr)

            for spks, fr, sem, spks1, fr1, sem1, spks2, fr2, sem2, spks3, fr3, sem3,  uid  in zip(centered[0], al[0], sems[0],centered[1], al[1], sems[1], centered[2], al[2], sems[2], centered[3], al[3], sems[3], units_id):
                fig, axes = plt.subplots(2, 4, sharex = 'all') #figsize= rozmiar figury
            
                plt.subplot(241)
                plt.eventplot(spks, linelengths=1.5, color="blue") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('both_ephys_alligned', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                a = plt.subplot(245)
                a
                plt.plot(t_vec, np.mean(fr, 0), color="blue")
                y1 = np.mean(fr, 0) + sem
                y2 = np.mean(fr, 0) - sem
                plt.fill_between(t_vec, y1, y2, alpha=0.5, zorder=2, color="blue")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.ylabel('trial', fontsize=4)
                plt.xlabel('time', fontsize=4)
                plt.ylabel('mean_fr', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(242)
                plt.eventplot(spks1, linelengths=1.5, color="green") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('both_n_ephys_alligned', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(246, sharey=a)
                plt.plot(t_vec, np.mean(fr1, 0), color="green")
                y3 = np.mean(fr1, 0) + sem1
                y4 = np.mean(fr1, 0) - sem1
                plt.fill_between(t_vec, y3, y4, alpha=0.5, zorder=2, color="green")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.xlabel('time', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(243)
                plt.eventplot(spks2, linelengths=1.5, color="orange") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('ephys_only', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(247, sharey=a)
                plt.plot(t_vec, np.mean(fr2, 0), color="orange")
                y5 = np.mean(fr2, 0) + sem2
                y6 = np.mean(fr2, 0) - sem2
                plt.fill_between(t_vec, y5, y6, alpha=0.5, zorder=2, color="orange")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.xlabel('time', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                
                plt.subplot(244)
                plt.eventplot(spks3, linelengths=1.5, color="red") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('non_ephys_only', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(248, sharey=a)
                plt.plot(t_vec, np.mean(fr3, 0), color="red")
                y7 = np.mean(fr3, 0) + sem3
                y8 = np.mean(fr3, 0) - sem3
                plt.fill_between(t_vec, y7, y8, alpha=0.5, zorder=2, color="red")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.xlabel('time', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                fig.savefig(behav_path + "\\" + str(uid) + '.png', facecolor="white", dpi = 250)
                fig.clf()
                plt.close("all")
                del spks, fr, spks1, fr1, spks2, fr2, uid
                gc.collect()

def old_plot_conditons_z_score(conditions, behaviors, tests, date_list): #!!!add creation of commined_plots_ephys dir!!! and rewrite to accomodate looping through conditons and plots without explicitly wrriting them
    # Function plotting rasters and psths in 2x4 figures based on required conditions (works per behavior)
    # 
    # conditions - list of str naming conditions from conditions_ready output (saved files)
    # behaviors - list of str naming behaviors used in the experiment
    # tests - list of str, output of list_exp_dirs
    # date_list - list of str, exp_dates    
    
    plt.ioff()
    for test, date in zip(tests, date_list):
        spks_ts, units_id = ephys.read_spikes(test, sampling_rate = 30000.0, read_only = "good")
        for behavior in behaviors:
            behav_path = test + "\\combined_plots_z-score\\" + behavior
            if not os.path.exists(behav_path):
                os.makedirs(behav_path)

            centered = []
            names = []
            sems = []
            al = []
            
            for condition in conditions:
                events_ts = np.loadtxt(test + "Conditions\\" + "camA_" + date + "_compr_" + behavior + "_" + condition + ".txt")                        
                centered_ts = ephys.calc_rasters(spks_ts, events_ts, pre_event = 5.0, post_event = 5.0)
                all_fr, mean_fr, sem_fr, t_vec = ephys.fr_events_binless(centered_ts, 0.2, trunc_gauss = 4,
                                                                    sampling_rate = 30000.0, sampling_out = 100,
                                                                    pre_event = 5.0, post_event = 5.0)
                all_zsc, mean_zsc, sem_zsc, bin_edges = ephys.zscore_events(all_fr, 0.01, pre_event = 5.0, post_event = 5.0)
                centered.append(centered_ts)
                names.append(condition)
                sems.append(sem_zsc)
                al.append(all_zsc)

            for spks, fr, sem, spks1, fr1, sem1, spks2, fr2, sem2, spks3, fr3, sem3,  uid  in zip(centered[0], al[0], sems[0],centered[1], al[1], sems[1], centered[2], al[2], sems[2], centered[3], al[3], sems[3], units_id):
                fig, axes = plt.subplots(2, 4, sharex = 'all', sharey='row') #figsize= rozmiar figury
                cmap = plt.get_cmap("tab10") # that's a color map
                axes[0, 0].tick_params(axis="both", labelsize=4)    
                axes[1, 0].tick_params(axis="both", labelsize=4)
                axes[0, 1].tick_params(axis="both", labelsize=4)
                axes[1, 1].tick_params(axis="both", labelsize=4)
                axes[0, 2].tick_params(axis="both", labelsize=4)
                axes[1, 2].tick_params(axis="both", labelsize=4)
                axes[0, 3].tick_params(axis="both", labelsize=4)
                axes[1, 3].tick_params(axis="both", labelsize=4)

                axes[0, 0].eventplot(spks, linelengths=1.5, color="blue") #docelowo w pętli po warunkach zamiast drugiego 0
                axes[1, 0].plot(t_vec, np.mean(fr, 0), color="blue")
                y1 = np.mean(fr, 0) + sem
                y2 = np.mean(fr, 0) - sem
                axes[1, 0].fill_between(t_vec, y1, y2, alpha=0.5, zorder=2, color="blue")
                axes[1, 0].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                axes[0, 0].set_title('both_ephys_alligned', fontsize=6)
                axes[0, 0].set_ylabel('trial', fontsize=4)
                axes[1, 0].set_xlabel('time', fontsize=4)
                axes[1, 0].set_ylabel('mean_fr', fontsize=4)

                axes[0, 1].eventplot(spks1, linelengths=1.5, color="green") #docelowo w pętli po warunkach zamiast drugiego 0
                axes[1, 1].plot(t_vec, np.mean(fr1, 0), color="green")
                y3 = np.mean(fr1, 0) + sem1
                y4 = np.mean(fr1, 0) - sem1
                axes[1, 1].fill_between(t_vec, y3, y4, alpha=0.5, zorder=2, color="green")
                axes[1, 1].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                axes[0, 1].set_title('both_n_ephys_alligned', fontsize=6)
                axes[1, 1].set_xlabel('time', fontsize=4)

                axes[0, 2].eventplot(spks2, linelengths=1.5, color="orange") #docelowo w pętli po warunkach zamiast drugiego 0
                axes[1, 2].plot(t_vec, np.mean(fr2, 0), color="orange")
                y5 = np.mean(fr2, 0) + sem2
                y6 = np.mean(fr2, 0) - sem2
                axes[1, 2].fill_between(t_vec, y5, y6, alpha=0.5, zorder=2, color="orange")
                axes[1, 2].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                axes[0, 2].set_title('ephys_only', fontsize=6)
                axes[1, 2].set_xlabel('time', fontsize=4)

                axes[0, 3].eventplot(spks3, linelengths=1.5, color="red") #docelowo w pętli po warunkach zamiast drugiego 0
                axes[1, 3].plot(t_vec, np.mean(fr3, 0), color="red")
                y7 = np.mean(fr3, 0) + sem3
                y8 = np.mean(fr3, 0) - sem3
                axes[1, 3].fill_between(t_vec, y7, y8, alpha=0.5, zorder=2, color="red")
                axes[1, 3].axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                axes[0, 3].set_title('non_ephys_only', fontsize=6)
                axes[1, 3].set_xlabel('time', fontsize=4)

                fig.savefig(behav_path + "\\" + str(uid) + '.png', facecolor="white", dpi = 250)
                fig.clf()
                plt.close("all")
                del spks, fr, spks1, fr1, spks2, fr2, uid
                gc.collect()

def plot_conditons_z_score(conditions, behaviors, tests, date_list): #!!!add creation of commined_plots_ephys dir!!! and rewrite to accomodate looping through conditons and plots without explicitly wrriting them
    # Function plotting rasters and psths in 2x4 figures based on required conditions (works per behavior)
    # 
    # conditions - list of str naming conditions from conditions_ready output (saved files)
    # behaviors - list of str naming behaviors used in the experiment
    # tests - list of str, output of list_exp_dirs
    # date_list - list of str, exp_dates    
    
    plt.ioff()
    for test, date in zip(tests, date_list):
        spks_ts, units_id = ephys.read_spikes(test, sampling_rate = 30000.0, read_only = "good")
        for behavior in behaviors:
            behav_path = test + "\\combined_plots_z-score\\" + behavior
            if not os.path.exists(behav_path):
                os.makedirs(behav_path)

            centered = []
            names = []
            sems = []
            al = []
            
            for condition in conditions:
                events_ts = np.loadtxt(test + "Conditions\\" + "camA_" + date + "_compr_" + behavior + "_" + condition + ".txt")                        
                centered_ts = ephys.calc_rasters(spks_ts, events_ts, pre_event = 5.0, post_event = 5.0)
                all_fr, mean_fr, sem_fr, t_vec = ephys.fr_events_binless(centered_ts, 0.2, trunc_gauss = 4,
                                                                    sampling_rate = 30000.0, sampling_out = 100,
                                                                    pre_event = 5.0, post_event = 5.0)
                all_zsc, mean_zsc, sem_zsc, bin_edges = ephys.zscore_events(all_fr, 0.01, pre_event = 5.0, post_event = 5.0)
                centered.append(centered_ts)
                names.append(condition)
                sems.append(sem_zsc)
                al.append(all_zsc)

            for spks, fr, sem, spks1, fr1, sem1, spks2, fr2, sem2, spks3, fr3, sem3,  uid  in zip(centered[0], al[0], sems[0],centered[1], al[1], sems[1], centered[2], al[2], sems[2], centered[3], al[3], sems[3], units_id):
                fig, axes = plt.subplots(2, 4, sharex = 'all') #figsize= rozmiar figury
            
                plt.subplot(241)
                plt.eventplot(spks, linelengths=1.5, color="blue") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('both_ephys_alligned', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                a = plt.subplot(245)
                a
                plt.plot(t_vec, np.mean(fr, 0), color="blue")
                y1 = np.mean(fr, 0) + sem
                y2 = np.mean(fr, 0) - sem
                plt.fill_between(t_vec, y1, y2, alpha=0.5, zorder=2, color="blue")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.ylabel('trial', fontsize=4)
                plt.xlabel('time', fontsize=4)
                plt.ylabel('z-scorel', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(242)
                plt.eventplot(spks1, linelengths=1.5, color="green") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('both_n_ephys_alligned', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(246, sharey=a)
                plt.plot(t_vec, np.mean(fr1, 0), color="green")
                y3 = np.mean(fr1, 0) + sem1
                y4 = np.mean(fr1, 0) - sem1
                plt.fill_between(t_vec, y3, y4, alpha=0.5, zorder=2, color="green")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.xlabel('time', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(243)
                plt.eventplot(spks2, linelengths=1.5, color="orange") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('ephys_only', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(247, sharey=a)
                plt.plot(t_vec, np.mean(fr2, 0), color="orange")
                y5 = np.mean(fr2, 0) + sem2
                y6 = np.mean(fr2, 0) - sem2
                plt.fill_between(t_vec, y5, y6, alpha=0.5, zorder=2, color="orange")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.xlabel('time', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                
                plt.subplot(244)
                plt.eventplot(spks3, linelengths=1.5, color="red") #docelowo w pętli po warunkach zamiast drugiego 0
                plt.title('non_ephys_only', fontsize=6)
                plt.tick_params(axis="both", labelsize=4)
                
                plt.subplot(248, sharey=a)
                plt.plot(t_vec, np.mean(fr3, 0), color="red")
                y7 = np.mean(fr3, 0) + sem3
                y8 = np.mean(fr3, 0) - sem3
                plt.fill_between(t_vec, y7, y8, alpha=0.5, zorder=2, color="red")
                plt.axvline(x = 0, linestyle = '--', color = 'gray', linewidth = 1)
                plt.xlabel('time', fontsize=4)
                plt.tick_params(axis="both", labelsize=4)
                
                fig.savefig(behav_path + "\\" + str(uid) + '.png', facecolor="white", dpi = 250)
                fig.clf()
                plt.close("all")
                del spks, fr, spks1, fr1, spks2, fr2, uid
                gc.collect()

def behavior_(file, df, framerate, video_len):
    behaviors = list(pd.unique(df.iloc[:,1]))

    dataframes = []

    for behavior in behaviors:
        behav = df.where(df["behavior"] == behavior)
        behav.dropna(inplace=True)
        behav.reset_index(drop=True, inplace=True)
        dataframes.append(behav)

    points = list(map(starts_stops, dataframes))

    behs = []
    for point in points:
        point = point.drop(["index", "stop"], axis=1)
        behs.append(point)

    behaviors = ["rearing", "grooming", "freezing", "social"]

    non_ephys_filled = pd.DataFrame(index = range(video_len), 
                        columns= [behavior + '_non_ephys'.format(behavior) for behavior in behaviors])
    
    ephys_filled = pd.DataFrame(index = range(video_len), 
                    columns= [behavior + '_ephys'.format(behavior) for behavior in behaviors])

    for behavior in behaviors:
        for point in points:
            if point["behavior"][0] == behavior + "_ephys":
                for i in range(len(point)):
                    name = pd.unique(point.iloc[:,2])
                    ephys_filled[name[0]][point["start"][i] - framerate:point["stop"][i]] = 1
            if point["behavior"][0] == behavior + "_non_ephys":
                for i in range(len(point)):
                    name = pd.unique(point.iloc[:,2])
                    non_ephys_filled[name[0]][point["start"][i] - framerate:point["stop"][i]] = 1

    non_ephys_filled = non_ephys_filled.fillna(0)
    ephys_filled = ephys_filled.fillna(0)
    
    return non_ephys_filled, ephys_filled