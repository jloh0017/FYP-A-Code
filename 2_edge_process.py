#%% 
import csv
import numpy as np
import pandas as pd
import math

def extract_feature_edge_attr_idx (data):
    window_edge_attr = np.zeros(data.shape, dtype = object)
    window_edge_idx = np.zeros(data.shape, dtype = object)
    window_feature = np.zeros(data.shape, dtype = object)
    #######################################################################################################
    for idx_window,ind_window in enumerate(data):
        ind_window_shape = ind_window.shape # get length of window

        # node features in every time frame in one window
        ind_window = (ind_window.T).reshape(ind_window_shape[2],ind_window_shape[1],ind_window_shape[0]) 
        # reshape each window to time frame, number of vehicles, features
        
        # edge attribute array for every time frame in one window
        time_frame_edge_attr = np.zeros(ind_window.shape[0], dtype = object) 

        # edge index array for every time frame in one window
        time_frame_edge_idx = np.zeros(ind_window.shape[0], dtype = object) 

        # construct the edge_attr and edge_idx matrix for each window
        for idx_time_fram,ind_time_frame in enumerate(ind_window):
            
            no_unique_vehicles = (ind_time_frame.shape)[0]
            edge_attr = np.zeros([no_unique_vehicles,no_unique_vehicles])
            edge_idx = np.zeros([no_unique_vehicles,no_unique_vehicles],dtype=int)
            # distance_matrix = np.zeros([no_unique_vehicles,no_unique_vehicles]) # for checking purposes

            # construct the edge_attr and edge_idx matrix for each time frame
            for idx_curr_vehicle,curr_vehicle_coord in enumerate(ind_time_frame):
                for idx_curr_surr_vehicle,curr_surr_vehicle in enumerate(ind_time_frame[idx_curr_vehicle:]):
                    v2v_diff = np.abs(np.subtract(curr_vehicle_coord,curr_surr_vehicle)) # vehicle to vehicle difference
                    euclid_dist = math.sqrt(pow(v2v_diff[0],2)+pow(v2v_diff[1],2))
                    edge_attr[idx_curr_vehicle][idx_curr_vehicle+idx_curr_surr_vehicle] = euclid_dist

                    cv_x = curr_vehicle_coord[0]
                    cv_y = curr_vehicle_coord[1]

                    csv_x = curr_surr_vehicle[0]
                    csv_y = curr_surr_vehicle[1]
                    exist_in_frame = cv_x > 0.04 and cv_y > 0.04 and csv_x > 0.04 and csv_y > 0.04
                    
                    #only collect adjacency if the 2 vehicles exist in the frame
                    # collect edge index if self or neighbour (this provided both cars are in the frame)
                    if exist_in_frame and ((idx_curr_vehicle == idx_curr_vehicle+idx_curr_surr_vehicle) or (euclid_dist > 0 and euclid_dist < 10)):
                        edge_idx[idx_curr_vehicle][idx_curr_vehicle+idx_curr_surr_vehicle] = 1
                        edge_attr[idx_curr_vehicle][idx_curr_vehicle+idx_curr_surr_vehicle] = euclid_dist
                    else:
                        edge_idx[idx_curr_vehicle][idx_curr_vehicle+idx_curr_surr_vehicle] = 0
                        edge_attr[idx_curr_vehicle][idx_curr_vehicle+idx_curr_surr_vehicle] = 0

            edge_attr = edge_attr+edge_attr.T-(np.diag(np.diag(edge_attr))) # reprocess the edge attr matrix
            edge_idx = edge_idx+edge_idx.T-(np.diag(np.diag(edge_idx))) # reprocess the edge idx matrix

            time_frame_edge_idx[idx_time_fram] = edge_idx
            time_frame_edge_attr[idx_time_fram] = edge_attr

        # compile every window's node features, edge attribute and edge index
        window_edge_attr[idx_window] = time_frame_edge_attr
        window_edge_idx[idx_window] = time_frame_edge_idx
        window_feature[idx_window] = ind_window

    return window_edge_attr,window_edge_idx,window_feature

vtds = ['0400-0415','0500-0515','0515-0530']
for scenario_idx,scenario in enumerate(vtds):
    descr = ['always_within_range']
    for criteria_idx,criteria in enumerate (descr):
        train_array = np.load(f"processed-trajectory-data\\1_train\\train_{scenario}.npy", allow_pickle = True)
        test_array = np.load(f"processed-trajectory-data\\2_test\\test_{scenario}.npy", allow_pickle = True)
        
        # edge attributes for every time frame in every window
        window_edge_attr_train = np.zeros(train_array.shape, dtype = object)
        window_edge_attr_test = np.zeros(test_array.shape, dtype = object) 

        # node features for every time frame in every window
        window_feature_train = np.zeros(train_array.shape, dtype = object)
        window_feature_test = np.zeros(test_array.shape, dtype = object)

        # edge index for every time frame in every window
        window_edge_idx_train = np.zeros(train_array.shape, dtype = object)
        window_edge_idx_test = np.zeros(test_array.shape, dtype = object)

        window_edge_attr_train,window_edge_idx_train,window_feature_train = extract_feature_edge_attr_idx(train_array)
        # save the numpy array for future use
        np.save(f"trajectory_data_for_graph_creation\\0_edge_idx\\train_set_{scenario}_edge_idx",window_edge_idx_train)
        np.save(f"trajectory_data_for_graph_creation\\1_edge_attr\\train_set_{scenario}_edge_attr",window_edge_attr_train)
        np.save(f"trajectory_data_for_graph_creation\\2_features\\train_set_{scenario}_features",window_feature_train)
        print("Finished Train Array Edge Processing")

        window_edge_attr_test,window_edge_idx_test,window_feature_test = extract_feature_edge_attr_idx(test_array)
        # save the numpy array for future use
        np.save(f"trajectory_data_for_graph_creation\\0_edge_idx\\test_set_{scenario}_edge_idx",window_edge_idx_test)
        np.save(f"trajectory_data_for_graph_creation\\1_edge_attr\\test_set_{scenario}_edge_attr",window_edge_attr_test)
        np.save(f"trajectory_data_for_graph_creation\\2_features\\test_set_{scenario}_features",window_feature_test)
        print("Finished Test Array Edge Processing")

        print(f"Finished {criteria}_{scenario}")
    print(f"Finished {scenario} processing")
    print("")




# slower but conlan7firm correct
        # for idx_window,ind_window in enumerate(test_array):
        #     ind_window_shape = ind_window.shape # orignal shape
        #     ind_window = (ind_window.T).reshape(ind_window_shape[2],ind_window_shape[1],ind_window_shape[0]) # reshape to time frame, number of vehicles, features
        #     time_frame_edge_attr = np.zeros(ind_window.shape[0], dtype = object)
        #     for idx_time_fram,ind_time_frame in enumerate(ind_window):
        #         # construct the edge_attr matrix
        #         no_unique_vehicles = (ind_time_frame.shape)[0]
        #         edge_attr = np.zeros([no_unique_vehicles,no_unique_vehicles])
        #         for idx_curr_vehicle,curr_vehicle_coord in enumerate(ind_time_frame):
        #             for idx_curr_surr_vehicle,curr_surr_vehicle in enumerate(ind_time_frame):
        #                 v2v_diff = np.abs(np.subtract(curr_vehicle_coord,curr_surr_vehicle)) # vehicle to vehcile difference
        #                 euclid_dist = math.sqrt(pow(v2v_diff[0],2)+pow(v2v_diff[1],2))
        #                 edge_attr[idx_curr_vehicle][idx_curr_surr_vehicle] = euclid_dist
        #         time_frame_edge_attr[idx_time_fram] = edge_attr
        #     window_edge_attr_test[idx_window] = time_frame_edge_attr
# %%
