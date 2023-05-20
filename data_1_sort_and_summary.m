%% 
clear all, clc, close all

paths = ['original-trajectory-data\trajectories-0400-0415.txt';...
    'original-trajectory-data\trajectories-0500-0515.txt';...
    'original-trajectory-data\trajectories-0515-0530.txt'];
sorted_paths = ['original-trajectory-data\trajectories-0400-0415-sorted.txt';...
    'original-trajectory-data\trajectories-0500-0515-sorted.txt';...
    'original-trajectory-data\trajectories-0515-0530-sorted.txt'];
scenario = ['0400-0415';'0500-0515';'0515-0530'];

for j = 1:size(paths,1)
    % get the current path
    current_path = paths(j,:);

    % read the table
    dataset = readtable(current_path);

    % get number of unique vehicles
    [unique_cars,ia,ic] = unique(dataset.Vehicle_ID);

    % get number of unique frames
    [unique_frames,ia,ic] = unique(dataset.Frame_ID);

    % get maximum number of cars that can appear in a frame
    [counts, frames] = groupcounts(dataset.Frame_ID);
    max_num_nodes_per_frame = max(counts)

    % get vehicle id with the most number of frames
    [highest_freq,highest_freq_index] = max(dataset.Total_Frames); % find which care appears the most

    % change from feet to metre
    dataset.Local_X = dataset.Local_X*0.3048;
    dataset.Local_Y = dataset.Local_Y*0.3048;

    % sort the dataset
    dataset_sorted = sortrows(dataset,["Frame_ID","Local_Y","Local_X","Vehicle_ID"],{'ascend'});

    % write the table
    writetable(dataset_sorted,sorted_paths(j,:),'Delimiter',' ') 

    % print
    fprintf('Scenario:%s, Num_unique_vehicles:%d, Num_unique_frames:%d, Car that appears the most:%d (%d times)\n',scenario(j,:),length(unique_cars),length(unique_frames),dataset.Vehicle_ID(highest_freq_index),highest_freq)
end

%% Check if the sorted dataset matches the original one
clear all, clc, close all
sorted_paths = ['original-trajectory-data\trajectories-0400-0415-sorted.txt';...
    'original-trajectory-data\trajectories-0500-0515-sorted.txt';...
    'original-trajectory-data\trajectories-0515-0530-sorted.txt'];
scenario = ['0400-0415';'0500-0515';'0515-0530'];

for j = 1:size(sorted_paths,1)
    current_path = sorted_paths(j,:);
    dataset = readtable(current_path);
    p = 0;
    [unique_cars,ia,ic] = unique(dataset.Vehicle_ID);
    [unique_frames,ia,ic] = unique(dataset.Frame_ID);
    [highest_freq,highest_freq_index] = max(dataset.Total_Frames); % find which care appears the most
    fprintf('Scenario:%s, Num_unique_vehicles:%d, Num_unique_frames:%d, Car that appears the most:%d (%d times)\n',scenario(j,:),length(unique_cars),length(unique_frames),dataset.Vehicle_ID(highest_freq_index),highest_freq)
end

% Scenario:0400-0415, Num_unique_vehicles:2052, Num_unique_frames:10013, Car that appears the most:3355 (1192 times)
% Scenario:0500-0515, Num_unique_vehicles:1836, Num_unique_frames:9787, Car that appears the most:2515 (2216 times)
% Scenario:0515-0530, Num_unique_vehicles:1790, Num_unique_frames:11685, Car that appears the most:2484 (2434 times)