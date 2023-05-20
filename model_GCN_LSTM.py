#%%
import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn import Linear
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import numpy as np
import random

INPUT_FRAMES = 50
DOWNSAMPLE_FREQ = 5
SAMPLING_INPUT_FRAMES = int(INPUT_FRAMES/DOWNSAMPLE_FREQ)
CURRENT_SAMPLING_FREQ = int(10/DOWNSAMPLE_FREQ)
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
					Linear(self.hidden_size, 2048),
					nn.ReLU(),
					nn.Linear(2048,4096),
					nn.ReLU(),
					nn.Linear(4096,2048),
					nn.ReLU(),
					Linear(2048,1024),
					nn.ReLU(),
					Linear(1024,512),
					nn.ReLU(),
					Linear(512,self.num_in_features*self.num_nodes)
				)

	def forward(self, feature_input,edge_indices_input,edge_indices_output,number_of_trajectories,stats):
		"""
		Forward pass
		"""
		# initial LSTM Variables
		hidden_state = torch.zeros(self.n_layer, number_of_trajectories, self.hidden_size).to(self.cur_device)
		cell_state = torch.zeros(self.n_layer, number_of_trajectories, self.hidden_size).to(self.cur_device)
		x_pred_encode = []
		y_pred_decode = []

		# collect the first one as the initial frame
		x_pred_encode.append(feature_input[0])

		# encode stage
		for i in range(SAMPLING_INPUT_FRAMES):
			input_feature = feature_input[i]
			input_edge_index = edge_indices_input[i]
			# print(torch.equal(input_edge_index,self.find_edge_index(input_feature,stats,number_of_trajectories)))
			if (i==SAMPLING_INPUT_FRAMES-1): # stop encoding at frame (INPUT_FRAMES-1)
				break

			# test_edge_index = self.find_edge_index(input_feature,stats,number_of_trajectories)
			# if (torch.equal(test_edge_index, input_edge_index)):
			# 	print("True")
			# else: 
			# 	print("False")

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
			# pass through GCN
			feature_output_gcn = self.GCNConv_cell(input_feature,input_edge_index)

			# reshape for LSTM - dim: [1,trajectory #,Number of GCN_output_features*Number of nodes]
			feature_output = torch.reshape(feature_output_gcn,(1,number_of_trajectories,self.num_out_features_GCN*self.num_nodes))

			# pass into LSTM and collect prediction
			output_features_lstm,(hidden_state,cell_state) = self.LSTM_cell(feature_output,(hidden_state,cell_state))
			cur_prediction = self.linear_relu_stack(output_features_lstm)
			cur_prediction = torch.reshape(cur_prediction,(number_of_trajectories*self.num_nodes,self.num_in_features))

			# collect the decoder predictions
			y_pred_decode.append(cur_prediction)

			# replace the next input_features to loop the output from the previous LSTM Cell
			input_feature = cur_prediction
			input_edge_index = self.find_edge_index(cur_prediction,stats,number_of_trajectories)

		return x_pred_encode,y_pred_decode
	
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
		input_features = torch.reshape(input_features,(number_of_trajectories,number_of_nodes,self.num_in_features))

		# dist = spatial.distance(input_features,input_features)
		# construct the edge_attr and edge_idx matrix for each trajectory
		for i in range(number_of_trajectories):
			cur_input_features = input_features[i]

			no_unique_vehicles = (cur_input_features.shape)[0]
			edge_attr = torch.zeros([no_unique_vehicles,no_unique_vehicles])
			edge_idx = torch.zeros([no_unique_vehicles,no_unique_vehicles],dtype=int)
			# distance_matrix = np.zeros([no_unique_vehicles,no_unique_vehicles]) # for checking purposes

			# construct the edge_attr and edge_idx matrix for each time frame
			edge_attr = torch.cdist(cur_input_features,cur_input_features)
			neighbour_threshold = 10
			edge_idx = (edge_attr <= neighbour_threshold).int()
			do_u_exist = torch.unsqueeze(torch.logical_and(cur_input_features[:,0]>0.04,cur_input_features[:,1]>0.04).int(),axis=0)
			do_u_exist_wif_me = torch.matmul((do_u_exist.T).cpu(),do_u_exist.cpu()).to(cur_device) # matmulgot problem for cuda
			no_i_dont_exist_wif_or_wifout_u = torch.logical_not(do_u_exist_wif_me).bool()
			edge_idx[no_i_dont_exist_wif_or_wifout_u] = 0
			edge_attr[no_i_dont_exist_wif_or_wifout_u] = 0
			
			# find the non-zero edge index
			edge_idx  = torch.nonzero(edge_idx)
			# extract the edge attributes of the non-zero edge index and reshape for GCN layer
			edge_idx = edge_idx.T.long()+i*number_of_nodes
			edge_index_all.append(edge_idx)


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
				
		return features_INPUT_FRAMES,edge_index_INPUT_FRAMES,features_OUTPUT_FRAMES,edge_index_OUTPUT_FRAMES
	
# %%
# Training Pipe 
# move to main.py after validation
from data_loader import *
cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize preliminary information and load data
torch.manual_seed(42)

BATCH_SIZE = 2
num_of_epochs = 401
train_loader,test_loader,train_dataset,test_dataset = load_data(batch_size=BATCH_SIZE,shuffle=False)
print(f"Length of train loader: {len(train_loader)}")
print(f"Length of test loader: {len(test_loader)}\n")

number_of_nodes = train_dataset.num_nodes
number_in_features = train_dataset.num_node_features
num_out_features_GCN = 2
hidden_size= 1024
stats_train = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
stats_val = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
# stats_train = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()] # for checking matrix transformations for unnormalized inputs
# stats_val = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()]
num_layers = 1

# initialize model and training criterion and optimizer
model_0 = GCN_LSTM(num_nodes=number_of_nodes,num_in_features=number_in_features,num_out_features_GCN=num_out_features_GCN, hidden_size=hidden_size, n_layer=num_layers,cur_device=cur_device, bias=True).to(cur_device)
criterion = torch.nn.MSELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)  # Define optimizer.
train_loss_arr = []
val_loss_arr = []
target_node = [0] #random.sample(range(100), 10)

for epoch in range(num_of_epochs):

	model_0.train()
	train_loss_epoch = 0 # find the average loss per epoch
	for i, batch_trajectories in enumerate(train_loader):
		# split batched trajectories into INPUT_FRAMES/OUTPUT_FRAMES and downsample
		batch_INPUT_FRAMES = batch_trajectories[:INPUT_FRAMES]
		batch_INPUT_FRAMES = batch_INPUT_FRAMES[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame
		batch_OUTPUT_FRAMES = batch_trajectories[INPUT_FRAMES:]
		batch_OUTPUT_FRAMES = batch_OUTPUT_FRAMES[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame
   
		# get the number of trajectories in the current batch_trajectories
		number_of_trajectories = int(batch_trajectories[0].num_nodes/number_of_nodes)
		
		# normalize input and reshape
		features_INPUT_FRAMES,edge_index_INPUT_FRAMES,features_OUTPUT_FRAMES,edge_index_OUTPUT_FRAMES = model_0.normalize(batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,stats_train)
		
		# perform the forward step
		optimizer.zero_grad()  # Clear gradients.
		x_pred,y_pred = model_0(feature_input=features_INPUT_FRAMES,edge_indices_input=edge_index_INPUT_FRAMES,edge_indices_output=edge_index_OUTPUT_FRAMES,number_of_trajectories=number_of_trajectories,stats=stats_train)

		# convert to torch for criterion
		x_pred = torch.stack(x_pred)
		features_INPUT_FRAMES = torch.stack(features_INPUT_FRAMES)
		y_pred = torch.stack(y_pred)
		features_OUTPUT_FRAMES = torch.stack(features_OUTPUT_FRAMES)

		# Compute the loss
		loss_encode = criterion(x_pred, features_INPUT_FRAMES)  
		loss_decode = criterion(y_pred, features_OUTPUT_FRAMES)  
		own_loss_encode = torch.sum(torch.pow(torch.abs(x_pred-features_INPUT_FRAMES),2))/(number_in_features*number_of_trajectories*number_of_nodes*SAMPLING_INPUT_FRAMES)
		own_loss_decode = torch.sum(torch.pow(torch.abs(y_pred-features_OUTPUT_FRAMES),2))/(number_in_features*number_of_trajectories*number_of_nodes*SAMPLING_OUTPUT_FRAMES)
		total_loss = loss_encode + loss_decode
		train_loss_epoch += total_loss
		own_total_loss =  torch.sqrt(own_loss_decode+own_loss_decode)
		# Derive gradients.
		total_loss.backward()  
		# Update parameters based on gradients.
		optimizer.step()  

		print(f"Number of trajectories in Train Batch Trajectories {i}: {number_of_trajectories}")
		print(f"Epoch: {epoch}, Train Batch Trajectories: {i} Training Encoder Loss: {loss_encode.item()}")
		print(f"Epoch: {epoch}, Train Batch Trajectories: {i} Training Decoder Loss: {loss_decode.item()}")
		print(f"Epoch: {epoch}, Train Batch Trajectories: {i} Training Total Loss: {total_loss.item()}\n")
		break
	train_loss_arr.append(train_loss_epoch.item()/(i+1))

	# start the validation phase
	val_loss_epoch = 0
	model_0.eval()
	with torch.no_grad():
		std_val, mean_val = stats_val[0].to(cur_device),stats_val[1].to(cur_device)
		for i, batch_trajectories_val in enumerate(train_loader):
			# split into INPUT_FRAMES/OUTPUT_FRAMES 
			batch_INPUT_FRAMES_val = batch_trajectories_val[:INPUT_FRAMES]
			batch_INPUT_FRAMES_val = batch_INPUT_FRAMES_val[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame
			batch_OUTPUT_FRAMES_val = batch_trajectories_val[INPUT_FRAMES:]
			batch_OUTPUT_FRAMES_val = batch_OUTPUT_FRAMES_val[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame

			# get the number of graphs in the current batched trajectory
			number_of_trajectories_val = int(batch_trajectories_val[0].num_nodes/number_of_nodes)
			
			# normalize and reshape for linear layer
			features_INPUT_FRAMES_val,edge_index_INPUT_FRAMES_val,features_OUTPUT_FRAMES_val,edge_index_OUTPUT_FRAMES_val = model_0.normalize(batch_INPUT_FRAMES_val,batch_OUTPUT_FRAMES_val,stats_val)

			# perform the forward step
			x_pred_val,y_pred_val = model_0(feature_input=features_INPUT_FRAMES_val,edge_indices_input=edge_index_INPUT_FRAMES_val,edge_indices_output=edge_index_OUTPUT_FRAMES_val,number_of_trajectories=number_of_trajectories_val,stats=stats_val)
			
			# convert to torch for criterion
			x_pred_val = torch.stack(x_pred_val)
			features_INPUT_FRAMES_val = torch.stack(features_INPUT_FRAMES_val)
			y_pred_val = torch.stack(y_pred_val)
			features_OUTPUT_FRAMES_val = torch.stack(features_OUTPUT_FRAMES_val)

			# loss function for the current batched trajectories
			loss_encode = criterion(x_pred_val, features_INPUT_FRAMES_val)  
			loss_decode = criterion(y_pred_val, features_OUTPUT_FRAMES_val)
			own_loss_encode = torch.sum(torch.pow(torch.abs(x_pred_val-features_INPUT_FRAMES_val),2))/(number_in_features*number_of_trajectories_val*number_of_nodes*SAMPLING_INPUT_FRAMES)
			own_loss_decode = torch.sum(torch.pow(torch.abs(y_pred_val-features_OUTPUT_FRAMES_val),2))/(number_in_features*number_of_trajectories_val*number_of_nodes*SAMPLING_OUTPUT_FRAMES)
			total_loss = loss_encode + loss_decode
			own_total_loss =  torch.sqrt(own_loss_decode+own_loss_decode)
			val_loss_epoch += total_loss

			print(f"Number of trajectories in Test Batch Trajectories {i}: {number_of_trajectories_val}")
			print(f"Epoch: {epoch}, Test Batch Trajectories: {i} Test Encoder Loss: {loss_encode.item()}")
			print(f"Epoch: {epoch}, Test Batch Trajectories: {i} Test Decoder Loss: {loss_decode.item()}")
			print(f"Epoch: {epoch}, Test Batch Trajectories: {i} Test Total Loss: {total_loss.item()}\n")

			# perform the testing once every 20 epochs
			if ((epoch)%50 == 0):
				for j in range(number_of_trajectories_val):
					# extract and unnormalize the nodes features for all frames for that trajectory
					cur_trajectory_feature = features_OUTPUT_FRAMES_val[:,j*number_of_nodes:(j+1)*number_of_nodes,:]*std_val+mean_val
					cur_trajectory_feature_pred = y_pred_val[:,j*number_of_nodes:(j+1)*number_of_nodes,:]*std_val+mean_val
					# downsample again to 1Hz and select the target node we want (include the very first starting point for the output)
					cur_trajectory_feature = torch.cat((torch.unsqueeze(cur_trajectory_feature[0,target_node,:],0),cur_trajectory_feature[CURRENT_SAMPLING_FREQ-1::CURRENT_SAMPLING_FREQ,target_node,:]),dim=0)
					cur_trajectory_feature_pred = torch.cat((torch.unsqueeze(cur_trajectory_feature_pred[0,target_node,:],0),cur_trajectory_feature_pred[CURRENT_SAMPLING_FREQ-1::CURRENT_SAMPLING_FREQ,target_node,:]),dim=0)

					# visualize the predictions
					plt.figure(figsize=(10,10))
					plt.subplot(1, 2, 1).set_xlim([0,5])
					plt.subplot(1, 2, 1).set_title("GT Trajectory")
					plt.plot(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
					plt.scatter(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
					plt.subplot(1, 2, 2).set_xlim([0,5])
					plt.subplot(1, 2, 2).set_title("Predicted Trajectory")
					plt.plot(cur_trajectory_feature_pred[:,:,0].cpu(),cur_trajectory_feature_pred[:,:,1].cpu())
					plt.scatter(cur_trajectory_feature_pred[:,:,0].cpu(),cur_trajectory_feature_pred[:,:,1].cpu())
					plt.show()

					plt.figure(figsize=(10,10))
					plt.xlim(0,40)
					plt.plot(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
					plt.plot(cur_trajectory_feature_pred[:,:,0].cpu(),cur_trajectory_feature_pred[:,:,1].cpu(),color= "red")
					plt.legend(["GT","Pred"])
					plt.show()

					# get the rmse for each second for some cars
					deviation = cur_trajectory_feature-cur_trajectory_feature_pred
					deviation_rmse = torch.sqrt(torch.sum(torch.pow(deviation,2),dim=2,keepdim=True))
					for k,each_sec in enumerate(deviation_rmse):
						print_string = f"Second: {k}"
						for h,each_car in enumerate(each_sec):
							print_string += f"|| Car {h+1} gt: {cur_trajectory_feature[k,h,:].cpu().numpy()} pred: {cur_trajectory_feature_pred[k,h,:].cpu().numpy()} deviation: {each_car.item():.4f} metres"
						print(print_string)
			break

	val_loss_arr.append(val_loss_epoch.item()/(i+1))

	# plot the losses
	if ((epoch)%50 == 0):
		plt.figure()
		# plt.ylim(0,1)
		plt.title("Train and Validation Loss per epoch")
		plt.plot(np.arange(0,len(train_loss_arr),1),train_loss_arr)
		plt.plot(np.arange(0,len(train_loss_arr),1),val_loss_arr)
		plt.legend(["train loss","validation loss"])
		plt.show()
	
	print("==========================================================================================")

# import scipy.io
# pred_np = y_pred_compare_unnormalized.cpu().numpy()
# gt_np = ground_truth_unnormalized.cpu().numpy()
# file_path = 'prediction_visualization\pred.mat'
# file_path2 = 'prediction_visualization\gt.mat'
# scipy.io.savemat(file_path, {'pred_np': pred_np})
# scipy.io.savemat(file_path2, {'gt_np': gt_np})

# # Create models directory 
# from pathlib import Path
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# # Create model save path 
# MODEL_NAME = "prediction_model_0.pkl"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# # Save the model state dict
# print(f"Saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
#            f=MODEL_SAVE_PATH) 
# %%
