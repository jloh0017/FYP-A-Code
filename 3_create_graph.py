#%%
import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data

def convert2graphs (features_window,edge_idx_window,edge_attr_window):
    """
	inputs
	  features_window: node features_window of all time frames in all windows of dataset
      edge_idx_window: edge index (adjacency) of all time frames in all windows of dataset

	outputs
	  graphs: a 2D array of graphs 
    """
        
	# Initiate empty container
    n_win = len(features_window)
    n_frame = len(features_window[0])

    Graph = np.empty([n_win,n_frame],dtype=object)
    count = 0
    for i in range(n_win):
        for j in range(n_frame):
            curr_frame_feature = torch.from_numpy(features_window[i][j])
            curr_frame_edge_idx = torch.from_numpy(edge_idx_window[i][j])
            curr_frame_edge_attr = torch.from_numpy(edge_attr_window[i][j])

            num_nodes = curr_frame_feature.shape[0]
            num_features_node = curr_frame_feature.shape[1]
            
            # find the non-zero edge index
            input_edge_idx  = torch.nonzero(curr_frame_edge_idx)
            # extract the edge attributes of the non-zero edge index and reshape
            input_edge_attr = curr_frame_edge_attr[input_edge_idx[:,0],input_edge_idx[:,1]]
            input_edge_attr = input_edge_attr.unsqueeze(-1).float()
            input_edge_idx = input_edge_idx.T.long()


            graph = Data(x=curr_frame_feature, edge_index=input_edge_idx, edge_attr=input_edge_attr,
            num_nodes=num_nodes, num_node_features=num_features_node)
            Graph[i,j] = graph
    return Graph

#%%
vtds = ['0400-0415','0500-0515','0515-0530']
for scenario_idx,scenario in enumerate(vtds):
    descr = ['always_within_range']
    for criteria_idx,criteria in enumerate (descr):
        window_edge_idx_train = np.load(f"trajectory_data_for_graph_creation\\0_edge_idx\\train_set_{scenario}_edge_idx.npy", allow_pickle = True)
        window_edge_attr_train  = np.load(f"trajectory_data_for_graph_creation\\1_edge_attr\\train_set_{scenario}_edge_attr.npy", allow_pickle = True)
        window_feature_train = np.load(f"trajectory_data_for_graph_creation\\2_features\\train_set_{scenario}_features.npy", allow_pickle = True)

        window_edge_idx_test = np.load(f"trajectory_data_for_graph_creation\\0_edge_idx\\test_set_{scenario}_edge_idx.npy", allow_pickle = True)
        window_edge_attr_test  = np.load(f"trajectory_data_for_graph_creation\\1_edge_attr\\test_set_{scenario}_edge_attr.npy", allow_pickle = True)
        window_feature_test = np.load(f"trajectory_data_for_graph_creation\\2_features\\test_set_{scenario}_features.npy", allow_pickle = True)

        t = torch.from_numpy(window_edge_attr_train[0][0])
        t1 = torch.from_numpy(window_edge_idx_train[0][0])
        t2 = torch.from_numpy(window_feature_train[0][0])
        train_graphs = convert2graphs(window_feature_train,window_edge_idx_train,window_edge_attr_train)
        test_graphs = convert2graphs(window_feature_test,window_edge_idx_test,window_edge_attr_test)
        print(f"Finished {criteria}_{scenario}")
        np.save(f"trajectory_data_for_graph_creation\\3_graphs\\train_graph_{scenario}",train_graphs)
        np.save(f"trajectory_data_for_graph_creation\\3_graphs\\test_graph_{scenario}",test_graphs)
    print(f"Finished {scenario} processing")
    print("")
# %%
