#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


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

def conditions_ready(file, df, TS, framerate, video_len):
    ###
    # Applies conditions and extracts instances when conditions are met 
    # file - path of analyzed file used for filename extraction
    # df - output of read_bvs function
    # TS - timestamps of frames relative to ephys recordings
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

    filename = get_filename(file)

    for behavior in behaviors:

        both = np.where((non_ephys_starts[behavior+"_non_ephys"] == 1) & (ephys_filled[behavior+"_ephys"] == 1))
        both = non_ephys_starts.iloc[both[0]].index
        np.savetxt(r"C:\Ephys_analysis\behawior_data\conditions\both_" + filename + behavior + ".txt", both)
        non_ephys_only = np.where((non_ephys_starts[behavior+"_non_ephys"] == 1) & (ephys_filled[behavior+"_ephys"] == 0))
        non_ephys_only = non_ephys_starts.iloc[non_ephys_only[0]].index
        np.savetxt(r"C:\Ephys_analysis\behawior_data\conditions\non_ephys_only_" + filename + behavior + ".txt", non_ephys_only)
        ephys_only = np.where((non_ephys_filled[behavior+"_non_ephys"] == 0) & (ephys_starts[behavior+"_ephys"] == 1))
        ephys_only = ephys_starts.iloc[ephys_only[0]].index
        np.savetxt(r"C:\Ephys_analysis\behawior_data\conditions\ephys_only_" + filename + behavior + ".txt", ephys_only)

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

def save_labels(filename, data, col_name):
    ###
    # Auxiliary function for label saving
    # filename - str, uses name of the input file
    # data - DataFrame, output of the simba_ready function
    #
    # Saves data into the same folder as original data as .csv file
    ###
    path = os.path.split(filename)
    filename = os.path.splitext(path[1])
    name = path[0] + "\\" + filename[0] + "_" + col_name + "_label.txt"
    return data.to_csv(name, index=False, header=False)    


### IN PROGRESS!!! ###
# Make directory for saving plots
save_plots = r"C:\Ephys_analysis"
nses = 6
sess_id = list[os.listdir(r"C:\Ephys_analysis")]
save_dir = save_plots + 'combined_plots_ephys\\'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Change some display options
plt.ioff() # don't display plots to save resources
#plt.tight_layout() # you can use it to make subplots closer to each other

# Loop through recordings (=sessions) 
for s in range(nses): 
    print('Session ' + str(s+1) + ' from ' + str(nses)) 
    # Make subfolder for storing plots from each of the sessions
    tmp_dir = save_dir + sess_ids[s] + '\\'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    # Loop through neurons
    for nrn in range(len(cntrd_ts[s])):
        # For each neuron make a new figure - here with 3 rows and 2 columns
        # You can control which subplots will have same limits - in this example y axis is shared between rows
        fig, axes = plt.subplots(3, 2, sharey = 'row') 
        cmap = plt.get_cmap("tab10") # that's a color map
        a = 0
        
        # For you it will be 3 behavior categories (own, partner, both)
        for cat in range(3):
            ymax = 0
                
                    # Select only trials belonging to a given category - you probably do it differently
                    tmp = [cntrd_ts[s][nrn][i] for i in odor_idxs] # the spikes (in list)
                    tmp2 = all_fr[s][nrn][odor_idxs,:] # the firing rates (in array)
                    
                    ##### Plotting itself #######
                        
                    # In row 0, column 'cat' plot raster plot from selected trials
                    # You won't need the lineoffsets argument
                    axes[0, cat].eventplot(tmp, color = cmap(a), lineoffsets = ypos, linewidths = 0.5)
                    
                    # In row 1 plot mean firing rate
                    axes[1, cat].plot(t_vec[s], np.mean(tmp2, 0), color = cmap(a), linewidth = 1)
                    tmp3 = np.mean(tmp2[:,pre_event*ifr_sr : pre_event*ifr_sr+hab_win*ifr_sr], 1) # habituation curve
                    
                    # In row 2 I plot something else, so you don't need it
                    axes[2, cat].plot(tmp3, "o-", markersize = 1.5, linewidth = 0.8, color = cmap(a))
                    
                    # I plot line to mark odor onset and offset, you can mark behavior onset
                    axes[0, cat].axvline(x = 0, linestyle = '-', color = 'gray', linewidth = 0.5)
                    axes[1, cat].axvline(x = 0, linestyle = '-', color = 'gray', linewidth = 0.5)
                    
                    
                    ####### End of plotting ##############
                    
                    #### Adjust visuals & set labels #### 
                    axes[0,0].set_title('Familiar odors', size = 8)
                    axes[0,1].set_title('Novel odors', size = 8)
                    
                    ## Two upper rows ##
                    axes[0,cat].tick_params(axis="both",direction="in", labelsize = 6)
                    axes[1,cat].xaxis.set_ticklabels([])
                    axes[1,cat].tick_params(axis="both",direction="in", top = True, labelsize = 6)
                    
                    axes[1,cat].set_xlabel("Time from odor [sec]", size = 6, labelpad = 0)
                    axes[0,0].set_ylabel("Trial", size = 6, labelpad = 0)
                    axes[1,0].set_ylabel("Mean firing rate", size = 6, labelpad = 0)
                    
                    ## Bottom row ##
                    axes[2,cat].tick_params(axis="both",direction="in", labelsize = 6)
                    axes[2,0].set_ylabel("Mean response", size = 6, labelpad = 0)
                    axes[2,cat].set_xlabel("Odor occurence", size = 6, labelpad = 0)

                    a = a + 1
        
        axes[0,0].sharex(axes[0,1]) # make some axes, but not all, common; I think you don't need it
        axes[1,0].sharex(axes[1,1])            
        fig.subplots_adjust(wspace=0.1, hspace=0.15) # reduce margins between subplots
        
        # Save the figure with neuron id in the title; you can adjust quality here
        fig.savefig(tmp_dir + str(spks_id[s][nrn]) + '.png', dpi = 250)
        plt.close(fig)
        
plt.ion()  # turn plots visibility back on

#Example code

def get_filename(file):
    filename = os.path.split(file)[1]
    filename = os.path.splitext(filename)[0]
    
    return filename

files50fps = load_files(r"E:\Neuropixel_data\Done\50fps", "bvs")
files30fps = load_files(r"E:\Neuropixel_data\Done\30fps", "bvs")
TS50fps = load_files(r"E:\Neuropixel_data\Done\50fps", "txt")
TS30fps = load_files(r"E:\Neuropixel_data\Done\30fps", "txt")

dataframes50fps = []
for file in files50fps:
    df50fps = read_bvs(file, 50)
    dataframes50fps.append(df50fps)

dataframes30fps = []
for file in files30fps:
    df30fps = read_bvs(file, 30)
    dataframes30fps.append(df30fps)

for file, TS, df in zip(files50fps, TS50fps, dataframes50fps):
    conditions_ready(file, df, TS, 50,  130000)



