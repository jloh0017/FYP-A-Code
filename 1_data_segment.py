#%%
import csv
import numpy as np
import pandas as pd
import math

vtds = ['0400-0415','0500-0515','0515-0530']

for j,scenario in enumerate(vtds):
    # extract how many unique vehicles and how many unique frames
    unique_vehicle_id    = []
    unique_vehicle_no    = 0
    unique_frame_id      = []
    unique_frame_no      = 0

    ptd01 = f'processed-trajectory-data\\0_within_range\\{scenario}.csv'
    descr = ['always_within_range']
    ptds = [ptd01]

    for index,ptd in enumerate(ptds):
        with open(ptd, 'r') as file:
            reader = csv.reader(file)
            row_num = 0
            for row in reader:
                if (row_num == 0):
                    row_num = row_num+1
                else:

                    row_num = row_num+1
                    current_vehicle_id = int(row[0])
                    current_frame_id = int(row[1])
                    
                    # get number of unique vehicles
                    # get number of unique frames
                    if current_vehicle_id not in unique_vehicle_id:
                        unique_vehicle_id.append(current_vehicle_id)
                        unique_vehicle_no = unique_vehicle_no+1

                    if current_frame_id not in unique_frame_id:
                        unique_frame_id.append(current_frame_id)
                        unique_frame_no = unique_frame_no+1

        # initialize the input representations 
        y_input = np.zeros([unique_vehicle_no, unique_frame_no])
        x_input = np.zeros([unique_vehicle_no, unique_frame_no])
        frame_id = np.zeros([unique_vehicle_no, unique_frame_no])
        vehicle_id = np.zeros([unique_vehicle_no, unique_frame_no])
        distance = np.zeros([unique_vehicle_no, unique_frame_no])

        # create the input representation
        with open(ptd, 'r') as file:
            reader = csv.reader(file)
            row_num = 0
            for row in reader:
                if (row_num == 0): # don't read header file
                    row_num = row_num+1
                else:
                    current_vehicle_id = int(row[0])
                    current_frame_id = int(row[1])
                    current_x_location = float(row[2])
                    current_y_location = float(row[3])

                    # check index of vehicle ID
                    vehicle_id_index = unique_vehicle_id.index(current_vehicle_id)
                    # check index of frame ID
                    frame_id_index = unique_frame_id.index(current_frame_id)

                    x_input[vehicle_id_index,frame_id_index] = current_x_location
                    y_input[vehicle_id_index,frame_id_index] = current_y_location
                    frame_id[vehicle_id_index,frame_id_index] = current_frame_id
                    vehicle_id[vehicle_id_index,frame_id_index] = current_vehicle_id
                    
        
        print(f"input representation finished, start partitioning for {scenario}: {descr[index]}")
        # start window partitioning
        number_of_trajectories = math.floor(unique_frame_no/80)

        window_dataset = np.zeros(number_of_trajectories, dtype = object) # window dataset array to hold all the windowes. Each window will have 
        start_idx_every_window = np.arange(0,unique_frame_no,80) # starting index for each individual window

        for i in range(number_of_trajectories):
            x_window = x_input[:,start_idx_every_window[i]:start_idx_every_window[i]+80]
            y_window = y_input[:,start_idx_every_window[i]:start_idx_every_window[i]+80]
            # feature = np.array([vh_id_window,x_window,y_window])
            feature = np.array([x_window,y_window],dtype = np.float64)
            window_dataset[i] = feature

        # example to demonstrate getting feature (1st = window, 2nd = feature, 3rd = for which unique vehicle, 4th = at what timeframe)
        p = window_dataset[0][0][0][0] 

        train_set_lim = math.floor(number_of_trajectories*(2/3))
        train_set = window_dataset[0:train_set_lim]
        test_set = window_dataset[train_set_lim:]
        
        # save the numpy array for future use
        np.save(f"processed-trajectory-data\\1_train\\train_{scenario}",train_set)
        np.save(f"processed-trajectory-data\\2_test\\test_{scenario}",test_set)

        unique_frame_id = np.array([unique_frame_id])
        unique_vehicle_id = np.transpose([np.array([0]+unique_vehicle_id)])
        print(f"data partitioning for {scenario}:{descr[index]} finished")
        x_input = np.concatenate((unique_frame_id, x_input))
        x_input = np.concatenate((unique_vehicle_id,x_input),axis=1)
        y_input = np.concatenate((unique_frame_id, y_input))
        y_input = np.concatenate((unique_vehicle_id,y_input),axis=1)
        # convert array into dataframe
        DF = pd.DataFrame(x_input)
        DF2 = pd.DataFrame(frame_id)
        DF3 = pd.DataFrame(vehicle_id)
        DF4 = pd.DataFrame(y_input)
        # save the dataframe as a csv file
        DF.to_csv(f"processed-trajectory-data\\3_check\\{scenario}_x_input.csv")
        DF4.to_csv(f"processed-trajectory-data\\3_check\\{scenario}_y_input.csv")

    print(f"Finished Processing Scenario: {scenario}")
    print("Unique Vehicle No.")
    print (unique_vehicle_no)
    print("Unique Frame No.")
    print (unique_frame_no)
    print("No. of windowes")
    print(number_of_trajectories)
    print('')

p = 0;
# %%
