#%%
import os
import time
import pickle
import dill
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

import matplotlib.pyplot as plt
from IPython.display import Audio, display, clear_output

class myDataset(Dataset):
	def __init__(self, data):
		self.data = data
		self.num_node_features = data[0,0].num_node_features
		self.num_nodes = data[0,0].num_nodes

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample

def padseq(batch):
	batch = np.asarray(batch).T
	if isinstance(batch[0,0], BaseData):
		new_batch = [Batch.from_data_list(graphs) for graphs in batch]
	return new_batch

def load_data(batch_size,shuffle):
	"""
	return 2D dataset and corresponding batch data loader
	- dataset is divided in [number of trajectories,number of frames], then loaded onto data loader
		- each dataset object is a graph with node features, edge index and edge attributes for one frame in one window (denotes trajectory)
	- dataloader batches the dataset based on number of trajectories per batch
	"""
    # Load data
	vtds = '0400-0415'

	train_graphs = np.load(f"trajectory_data_for_graph_creation\\3_graphs\\train_graph_{vtds}.npy",allow_pickle = True)
	
	test_graphs = np.load(f"trajectory_data_for_graph_creation\\3_graphs\\test_graph_{vtds}.npy",allow_pickle = True)

	train_dataset = myDataset(train_graphs)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,collate_fn= padseq, pin_memory=True)

	test_dataset = myDataset(test_graphs)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,collate_fn= padseq, pin_memory=True)
	
	return train_loader,test_loader,train_dataset,test_dataset

def find_mean_and_std(train_or_val,batch_size):
	"""
	To find the mean and standard deviation of input data

	for training data, find mean and std of all features of all frames in all training batches

	for testing data, find mean and std of all features in first 30 frames in all testing batches
	"""
	train_loader,test_loader,train_dataset,test_dataset = load_data(batch_size=batch_size,shuffle=False)

	if train_or_val: # get mean and std of all frames in all windows
		features_all_nodes_all_frames_all_windows = []
		for i, batch_trajectories in enumerate(train_loader):
			
			for j, frame_batched_trajectories in enumerate(batch_trajectories):
				# collect all the features of all nodes in each frame of the batched trajectories
				if (j == 0):
					features_all_nodes_all_frames = frame_batched_trajectories.x
				else:
					test = frame_batched_trajectories.x
					features_all_nodes_all_frames = torch.cat((features_all_nodes_all_frames,test),0)

			# collect all the features of all nodes in all frames for the current batched trajectories
			if(i == 0):
				features_all_nodes_all_frames_all_windows = features_all_nodes_all_frames
			else:
				features_all_nodes_all_frames_all_windows = torch.cat((features_all_nodes_all_frames_all_windows,features_all_nodes_all_frames),0)
			
	else:
		features_all_nodes_all_frames_all_windows = []
		for i, batch_trajectories in enumerate(test_loader):
			for j, frame_batched_trajectories in enumerate(batch_trajectories[:30]):
				# collect all the features of all nodes in each frame of the first 30 frames of the batched trajectories
				if (j == 0):
					features_all_nodes_all_frames = frame_batched_trajectories.x
				else:
					test = frame_batched_trajectories.x
					features_all_nodes_all_frames = torch.cat((features_all_nodes_all_frames,test),0)	

			# collect all the features of all nodes in all frames for the current batched trajectories
			if(i == 0):
				features_all_nodes_all_frames_all_windows = features_all_nodes_all_frames
			else:
				features_all_nodes_all_frames_all_windows = torch.cat((features_all_nodes_all_frames_all_windows,features_all_nodes_all_frames),0)

	# find the mean and standard deviation of non-zero features
	non_zero_index = torch.nonzero(features_all_nodes_all_frames_all_windows)
	non_zero_features_input = features_all_nodes_all_frames_all_windows[non_zero_index[:,0],non_zero_index[:,1]].reshape(-1,2)
	std,mean = torch.std_mean(non_zero_features_input, dim=0)
	std = std.float()
	mean = mean.float()
	
	return[std,mean]
# %%
