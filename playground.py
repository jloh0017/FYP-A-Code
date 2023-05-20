#%%
import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn import Linear
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import numpy as np
from data_loader import *

cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_FRAMES = 50
DOWNSAMPLE_FREQ = 10
SAMPLING_INPUT_FRAMES = int(INPUT_FRAMES/DOWNSAMPLE_FREQ)
OUTPUT_FRAMES = 80-INPUT_FRAMES
SAMPLING_OUTPUT_FRAMES = int(OUTPUT_FRAMES/DOWNSAMPLE_FREQ)

class GCN_LSTM(nn.Module):
	def __init__(self, num_nodes=0,num_in_features=0,num_out_features_GCN=0, hidden_size=0, n_layer=1,cur_device='cpu', bias=True):
		super().__init__()
		self.n_layer = n_layer
		self.num_nodes = num_nodes
		self.num_in_features = num_in_features
		self.num_out_features_GCN = num_out_features_GCN 
		self.hidden_size = hidden_size
		self.cur_device = cur_device

		# layer
		self.GCNConv_cell = GCNConv(in_channels = num_in_features,out_channels=num_out_features_GCN)
		self.LSTM_cell = LSTM(input_size=num_out_features_GCN*num_nodes,hidden_size=hidden_size,num_layers=n_layer)
		self.linear_relu_stack = nn.Sequential(
					Linear(self.hidden_size, 512),
					nn.ReLU(),
					nn.Linear(512, 1028),
					nn.ReLU(),
					nn.Linear(1028, 512),
					nn.ReLU(),
					Linear(512,self.num_in_features*num_nodes)
				)
		self.GATConv_cell = GATConv(in_channels = num_in_features,out_channels=num_out_features_GCN)

	def forward(self, feature_input,edge_index,batch_index,number_of_trajectories,stats):
		"""
		Forward pass
		"""
		# initial LSTM Variables
		hidden_state = torch.zeros(self.n_layer, number_of_trajectories, self.hidden_size).to(self.cur_device)
		cell_state = torch.zeros(self.n_layer, number_of_trajectories, self.hidden_size).to(self.cur_device)
		x_pred_encode = []
		x_pred_decode = []

		# collect the first one as the initial frame
		x_pred_encode.append(feature_input[0])

		# encode stage
		for i in range(SAMPLING_INPUT_FRAMES):
			input_feature = feature_input[i]
			input_edge_index = edge_index[i]
			input_batch_index = batch_index[i]

			if (i==SAMPLING_INPUT_FRAMES-1): # stop encoding at frame (INPUT_FRAMES-1)
				break

			# pass through GCN
			output_feature_gcn = self.GCNConv_cell(input_feature,input_edge_index)

			# reshape for LSTM - dim: [1,trajectory #,Number of GCN_output_features*Number of nodes]
			output_feature_gcn= torch.reshape(output_feature_gcn,(1,number_of_trajectories,self.num_out_features_GCN*self.num_nodes))

			# pass into LSTM and collect prediction
			output_feature_lstm,(hidden_state,cell_state) = self.LSTM_cell(output_feature_gcn,(hidden_state,cell_state))
			cur_prediction = self.linear_relu_stack(output_feature_lstm)
			cur_prediction = torch.reshape(cur_prediction,(number_of_trajectories*self.num_nodes,self.num_in_features))
			x_pred_encode.append(cur_prediction)

		# decode stage
		for i in range(SAMPLING_OUTPUT_FRAMES):
			print(i)
			# pass through GCN
			feature_output_gcn = self.GCNConv_cell(input_feature,input_edge_index)

			# reshape for LSTM - dim: [1,trajectory #,Number of GCN_output_features*Number of nodes]
			feature_output = torch.reshape(feature_output_gcn,(1,number_of_trajectories,self.num_out_features_GCN*self.num_nodes))

			# pass into LSTM and collect prediction
			output_features_lstm,(hidden_state,cell_state) = self.LSTM_cell(feature_output,(hidden_state,cell_state))
			cur_prediction = self.linear_relu_stack(output_features_lstm)
			cur_prediction = torch.reshape(cur_prediction,(number_of_trajectories*self.num_nodes,self.num_in_features))

			# collect the decoder predictions
			x_pred_decode.append(cur_prediction)

			# replace the next input_features to loop the output from the previous LSTM Cell
			input_feature = cur_prediction
			input_edge_index = self.find_edge_index(cur_prediction,stats,number_of_trajectories)

		return x_pred_encode,x_pred_decode
	
	def find_edge_index(self,input_features,stats,number_of_trajectories):
		"""
		creates/predicts the adjacency matrix for the GCN in the encoder and decoder phase
		==> returns a 2 row adjancency for all trajectories in the batch
		"""
		# init the variables
		edge_index_all = []
		std, mean = stats[0].to(cur_device),stats[1].to(cur_device)

		# unnormalize the prediction
		input_features = input_features*std+mean
		input_features = torch.reshape(input_features,(number_of_trajectories,number_of_nodes,number_of_features))

		# construct the edge_attr and edge_idx matrix for each trajectory
		for i in range(number_of_trajectories):
			cur_input_features = input_features[i]
			edge_index_idv = torch.zeros(self.num_nodes*number_of_trajectories,self.num_nodes*number_of_trajectories)
			for idx_curr_vehicle,curr_vehicle_coord in enumerate(cur_input_features):
				for idx_curr_surr_vehicle,curr_surr_vehicle in enumerate(cur_input_features[idx_curr_vehicle:]):
					v2v_diff = torch.abs(torch.subtract(curr_vehicle_coord,curr_surr_vehicle)) # vehicle to vehicle difference
					euclid_dist = torch.sqrt(torch.pow(v2v_diff[0],2)+torch.pow(v2v_diff[1],2))

					cv_x = curr_vehicle_coord[0]
					cv_y = curr_vehicle_coord[1]
					csv_x = curr_surr_vehicle[0]
					csv_y = curr_surr_vehicle[1]
					exist_in_frame = cv_x > 0.04 and cv_y > 0.04 and csv_x > 0.04 and csv_y > 0.04
					
					#only collect adjacency if the 2 vehicles exist in the frame
					# collect edge index if self or neighbour (this provided both cars are in the frame)
					if exist_in_frame and ((idx_curr_vehicle == idx_curr_vehicle+idx_curr_surr_vehicle) or (euclid_dist > 0 and euclid_dist < 10)):
						edge_index_idv[idx_curr_vehicle][idx_curr_vehicle+idx_curr_surr_vehicle] = 1
					else:
						edge_index_idv[idx_curr_vehicle][idx_curr_vehicle+idx_curr_surr_vehicle] = 0

			edge_index_idv = edge_index_idv+edge_index_idv.T-(np.diag(np.diag(edge_index_idv))) # reprocess the edge idx matrix
			
			# find the non-zero edge index
			edge_index_idv  = torch.nonzero(edge_index_idv)
			# extract the edge attributes of the non-zero edge index and reshape for GCN layer
			edge_index_idv = edge_index_idv.T.long()+i*number_of_nodes
			edge_index_all.append(edge_index_idv)

		return torch.cat(edge_index_all, dim=1).to(self.cur_device)
	
	def normalize(self,batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,stats):
		std,mean = stats
		"""
		returns normalized
		features_INPUT/OUTPUT_FRAMES: list of (# of input/output frames) tensors with dim: (number of trajectories*number of nodes) x number of features
		edge_index_INPUT/OUTPUT_FRAMES: list of (# of input/output frames) edge index tensors with dim: 2 x (number of nodes present in each adjancency for # of trajectories)
		batch_index_INPUT/OUTPUT_FRAMES: list of (# of input/output frames) batch index tensors with dim; 1 x (number of nodes present in each adjancency for # of trajectories)
		"""
		
		features_INPUT_FRAMES = []
		edge_index_INPUT_FRAMES = []
		edge_attri_INPUT_FRAMES = []
		batch_index_INPUT_FRAMES = []
		for j, frame_batched_trajectories in enumerate(batch_INPUT_FRAMES):
			current_frame_feature = frame_batched_trajectories.x

			# normalize data first
			normalized_frame_feature = (current_frame_feature-mean)/std

			features_INPUT_FRAMES.append((normalized_frame_feature.float()).to(self.cur_device))
			edge_index_INPUT_FRAMES.append(frame_batched_trajectories.edge_index.to(self.cur_device))
			edge_attri_INPUT_FRAMES.append(frame_batched_trajectories.edge_attr.to(self.cur_device))
			batch_index_INPUT_FRAMES.append(frame_batched_trajectories.batch.to(self.cur_device))
		
		features_OUTPUT_FRAMES = []
		edge_index_OUTPUT_FRAMES = []
		edge_attri_OUTPUT_FRAMES = []
		batch_index_OUTPUT_FRAMES = []
		for j, frame_batched_trajectories in enumerate(batch_OUTPUT_FRAMES):
			current_frame_feature = frame_batched_trajectories.x

			# normalize data first
			normalized_frame_feature = (current_frame_feature-mean)/std

			features_OUTPUT_FRAMES.append((normalized_frame_feature.float()).to(self.cur_device))
			edge_index_OUTPUT_FRAMES.append(frame_batched_trajectories.edge_index.to(self.cur_device))
			edge_attri_OUTPUT_FRAMES.append(frame_batched_trajectories.edge_attr.to(self.cur_device))
			batch_index_OUTPUT_FRAMES.append(frame_batched_trajectories.batch.to(self.cur_device))
				
		return features_INPUT_FRAMES,edge_index_INPUT_FRAMES,batch_index_INPUT_FRAMES,features_OUTPUT_FRAMES,edge_index_OUTPUT_FRAMES,batch_index_OUTPUT_FRAMES

BATCH_SIZE = 3
num_of_epochs = 300
train_loader,test_loader,train_dataset,test_dataset = load_data(batch_size=BATCH_SIZE,shuffle=False)
print(f"Length of train loader: {len(train_loader)}")
print(f"Length of test loader: {len(test_loader)}\n")

number_of_nodes = train_dataset.num_nodes
number_of_features = train_dataset.num_node_features
hidden_size= 512
stats_train = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
stats_val = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
stats_train = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()] # for checking matrix transformations for unnormalized inputs
stats_val = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()]
num_layers = 2

model_0 = GCN_LSTM(num_nodes=number_of_nodes,num_in_features=number_of_features,num_out_features_GCN=number_of_features, hidden_size=hidden_size, n_layer=num_layers,cur_device=cur_device, bias=True).to(cur_device)

# %%
for i, batch_trajectories in enumerate(train_loader):
	# split batched trajectories into INPUT_FRAMES/OUTPUT_FRAMES and downsample
	batch_INPUT_FRAMES = batch_trajectories[:INPUT_FRAMES]
	batch_INPUT_FRAMES = batch_INPUT_FRAMES[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame
	batch_OUTPUT_FRAMES = batch_trajectories[INPUT_FRAMES:]
	batch_OUTPUT_FRAMES = batch_OUTPUT_FRAMES[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame
    # get the number of trajectories in the current batch_trajectories
	number_of_trajectories = int(batch_trajectories[0].num_nodes/number_of_nodes)

    # normalize input and reshape
	features_INPUT_FRAMES,edge_index_INPUT_FRAMES,batch_index_INPUT_FRAMES,features_OUTPUT_FRAMES,edge_index_OUTPUT_FRAMES,batch_index_OUTPUT_FRAMES = model_0.normalize(batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,stats_train)
	break

x_pred,y_pred = model_0(feature_input=features_INPUT_FRAMES,edge_index=edge_index_INPUT_FRAMES,batch_index=batch_index_INPUT_FRAMES,number_of_trajectories=number_of_trajectories,stats=stats_train)

# %%
x_pred_encode = torch.cat(x_pred, dim=0)
features_INPUT_FRAMES_test = torch.cat(features_INPUT_FRAMES, dim=0)

y_pred_decode = torch.cat(y_pred, dim=0)
features_OUTPUT_FRAMES_test = torch.cat(features_OUTPUT_FRAMES, dim=0)

criterion = torch.nn.MSELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.01)  # Define optimizer.

loss_encode = criterion(x_pred_encode, features_INPUT_FRAMES_test)  
loss_decode = criterion(y_pred_decode, features_OUTPUT_FRAMES_test)  # Compute the loss
# %%
