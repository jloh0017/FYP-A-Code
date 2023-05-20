#%%
import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn import Linear
from torch.nn import Dropout
import matplotlib.pyplot as plt
import numpy as np

INPUT_FRAMES = 50
DOWNSAMPLE_FREQ = 5
CURRENT_SAMPLING_FREQ = int(10/DOWNSAMPLE_FREQ)
SAMPLING_INPUT_FRAMES = int(INPUT_FRAMES/DOWNSAMPLE_FREQ)
OUTPUT_FRAMES = 80-INPUT_FRAMES
SAMPLING_OUTPUT_FRAMES = int(OUTPUT_FRAMES/DOWNSAMPLE_FREQ)

class LSTM_EnDecoder(nn.Module):
	def __init__(self, num_nodes=0,num_features=0,num_out_features=0, n_layer=1,cur_device='cpu', bias=True):
		super().__init__()
		self.n_layer = n_layer
		self.num_nodes = num_nodes
		self.num_in_features = num_features*num_nodes
		self.num_out_features = num_out_features 
		self.cur_device = cur_device

		# layer
		self.LSTM_cell = LSTM(input_size=self.num_in_features,hidden_size=self.num_out_features,num_layers=self.n_layer)
		self.linear_relu_stack = nn.Sequential(
					Linear(self.num_out_features, 2048),
					nn.ReLU(),
					nn.Linear(2048,4096),
					nn.ReLU(),
					nn.Linear(4096,2048),
					nn.ReLU(),
					Linear(2048,1024),
					nn.ReLU(),
					Linear(1024,512),
					nn.ReLU(),
					Linear(512,self.num_in_features)
				)
		self.dropout = Dropout(p=0.1)

	def forward(self, input_frames,number_of_trajectories):
		# Define LSTM_init inputs
		init_input = torch.zeros(1,number_of_trajectories,self.num_in_features).to(self.cur_device)
		hidden_state = torch.zeros(self.n_layer, number_of_trajectories, self.num_out_features).to(self.cur_device)
		cell_state = torch.zeros(self.n_layer, number_of_trajectories, self.num_out_features).to(self.cur_device)
		x_pred_encode = []
		x_pred_decode = []

		# initial forward pass
		output,(hidden_state,cell_state) = self.LSTM_cell(init_input,(hidden_state,cell_state))
		cur_prediction = self.linear_relu_stack(output)
		k = 0

		# predict x_0
		x_pred_encode.append(cur_prediction)
		# x_pred_encode.append(input_frames[0].unsqueeze(dim=0)) # for checking
		k = k+1

		# Encode Stage (x_1 ==> x_3)
		for i,idv_frame in enumerate(input_frames):
			if (i==SAMPLING_INPUT_FRAMES-1): # 
				break
			output_frame,(hidden_state,cell_state) = self.LSTM_cell(idv_frame.unsqueeze(dim=0),(hidden_state,cell_state))
			cur_prediction = self.linear_relu_stack(output_frame)
			# collect the encoder predictions (represent the INPUT_FRAMES frames)
			# x_pred_encode.append(input_frames[i+1].unsqueeze(dim=0)) # for checking
			x_pred_encode.append(cur_prediction)
			k = k+1
		# change the encoder predictions to tensor
		x_pred_encode = torch.cat(x_pred_encode, dim=0)

		# Decode Stage (x_4 ==> x_7)
		idv_frame = idv_frame.unsqueeze(dim=0)
		for i in range(SAMPLING_OUTPUT_FRAMES):
			output_frame,(hidden_state,cell_state) = self.LSTM_cell(idv_frame,(hidden_state,cell_state))
			# output_frame = self.dropout(output_frame)
			cur_prediction = self.linear_relu_stack(output_frame)
			# collect the decoder predictions (represent the OUTPUT_FRAMES)
			x_pred_decode.append(cur_prediction)
			k = k+1
			# replace the next idv frame to loop the output from the previous LSTM Cell
			idv_frame = cur_prediction
		
		x_pred_decode = torch.cat(x_pred_decode, dim=0)
		return x_pred_encode,x_pred_decode
	
	def normalize(self,batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,number_of_features,number_of_nodes,number_of_trajectories,stats):
		std,mean = stats
		"""
		returns normalized
		features_INPUT_FRAMES: input feature tensor for that current batched trajectory
			Dim: [frames,trajectory #,Number of nodes*number of features]
			trajectory #: details which trajectory in the current batched trajectories
			features: x,y features 
		features_OUTPUT_FRAMES: output feature tensor for that current batched trajectory
			Dim: [frames,trajectory #,Number of nodes*number of features]
			trajectory #: details which trajectory in the current batched trajectories
			features: x,y features 
			frames: details which time frame
		"""
		# normalize data first
		# reshape into frames x batch x (num nodes * num features)
		features_INPUT_FRAMES = []
		for j, frame_batched_trajectories in enumerate(batch_INPUT_FRAMES):
			current_frame_feature = frame_batched_trajectories.x
			normalized_frame_feature = (current_frame_feature-mean)/std
			if j == 0:
				features_INPUT_FRAMES = torch.reshape(normalized_frame_feature,(1,number_of_trajectories,number_of_features*number_of_nodes))
			else:
				test = torch.reshape(normalized_frame_feature,(1,number_of_trajectories,number_of_features*number_of_nodes))
				features_INPUT_FRAMES = torch.cat((features_INPUT_FRAMES,test),dim=0)
		
		features_OUTPUT_FRAMES = []
		for j, frame_batched_trajectories in enumerate(batch_OUTPUT_FRAMES):
			current_frame_feature = frame_batched_trajectories.x
			normalized_frame_feature = (current_frame_feature-mean)/std

			if j == 0:
				features_OUTPUT_FRAMES = torch.reshape(normalized_frame_feature,(1,number_of_trajectories,number_of_features*number_of_nodes))
			else:
				test = torch.reshape(normalized_frame_feature,(1,number_of_trajectories,number_of_features*number_of_nodes))
				features_OUTPUT_FRAMES = torch.cat((features_OUTPUT_FRAMES,test),dim=0)
				
		return (features_INPUT_FRAMES.float()).to(self.cur_device),(features_OUTPUT_FRAMES.float()).to(self.cur_device)
	
# %%
# Training Pipe 
# move to main.py after validation
from data_loader import *
cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize preliminary information and load data
torch.manual_seed(42)

BATCH_SIZE = 2
num_of_epochs = 401
shuffle = False
train_loader,test_loader,train_dataset,test_dataset = load_data(batch_size=BATCH_SIZE,shuffle=shuffle)
print(f"Length of train loader: {len(train_loader)}")
print(f"Length of test loader: {len(test_loader)}\n")

number_of_nodes = train_dataset.num_nodes
number_of_features = train_dataset.num_node_features
number_of_out_features = 1024
stats_train = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
stats_val = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
# stats_train = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()] # for checking matrix transformations for unnormalized inputs
# stats_val = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()]
num_layers = 1

# initialize model and training criterion and optimizer
model_0 = LSTM_EnDecoder(num_nodes=number_of_nodes,num_features=number_of_features,num_out_features=number_of_out_features,n_layer=num_layers,cur_device=cur_device).to(cur_device)
criterion = torch.nn.MSELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)  # Define optimizer.
train_loss_arr = []
val_loss_arr = []
target_node = [0]

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
		features_INPUT_FRAMES,features_OUTPUT_FRAMES = model_0.normalize(batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,number_of_features,
									number_of_nodes,number_of_trajectories,stats_train)
		
		# perform the forward step
		optimizer.zero_grad()  # Clear gradients.
		x_pred,y_pred = model_0(features_INPUT_FRAMES,number_of_trajectories)

		# Compute the loss
		loss_encode = criterion(x_pred, features_INPUT_FRAMES)  
		loss_decode = criterion(y_pred, features_OUTPUT_FRAMES)  
		own_loss_encode = torch.sum(torch.pow(torch.abs(x_pred-features_INPUT_FRAMES),2))/(number_of_features*number_of_trajectories*number_of_nodes*SAMPLING_INPUT_FRAMES)
		own_loss_decode = torch.sum(torch.pow(torch.abs(y_pred-features_OUTPUT_FRAMES),2))/(number_of_features*number_of_trajectories*number_of_nodes*SAMPLING_OUTPUT_FRAMES)
		total_loss = loss_encode + loss_decode
		train_loss_epoch += total_loss
		# Derive gradients.
		total_loss.backward()  
		# Update parameters based on gradients.
		optimizer.step()  

		print(f"Number of trajectories in Train Batch Trajectories {i}: {number_of_trajectories}")
		print(f"Epoch: {epoch+1}, Train Batch Trajectories: {i} Training Encoder Loss: {loss_encode.item()}")
		print(f"Epoch: {epoch+1}, Train Batch Trajectories: {i} Training Decoder Loss: {loss_decode.item()}")
		print(f"Epoch: {epoch+1}, Train Batch Trajectories: {i} Training Total Loss: {total_loss.item()}\n")
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
			features_INPUT_FRAMES_val,features_OUTPUT_FRAMES_val = model_0.normalize(batch_INPUT_FRAMES_val,batch_OUTPUT_FRAMES_val,number_of_features,
										number_of_nodes,number_of_trajectories_val,stats_val)

			# perform the forward step
			x_pred_val,y_pred_val = model_0(features_INPUT_FRAMES_val,number_of_trajectories_val)
			
			# loss function for the current batched trajectories
			loss_encode = criterion(x_pred_val, features_INPUT_FRAMES_val)  
			loss_decode = criterion(y_pred_val, features_OUTPUT_FRAMES_val) 
			own_loss_encode = torch.sum(torch.pow(torch.abs(x_pred_val-features_INPUT_FRAMES_val),2))/(number_of_features*number_of_trajectories_val*number_of_nodes*SAMPLING_INPUT_FRAMES)
			own_loss_decode = torch.sum(torch.pow(torch.abs(y_pred_val-features_OUTPUT_FRAMES_val),2))/(number_of_features*number_of_trajectories_val*number_of_nodes*SAMPLING_OUTPUT_FRAMES)
			total_loss = loss_encode + loss_decode
			own_total_loss =  own_loss_decode+own_loss_decode
			val_loss_epoch += total_loss
			print(f"Number of trajectories in Test Batch Trajectories {i}: {number_of_trajectories_val}")
			print(f"Epoch: {epoch+1}, Test Batch Trajectories: {i} Test Encoder Loss: {loss_encode.item()}")
			print(f"Epoch: {epoch+1}, Test Batch Trajectories: {i} Test Decoder Loss: {loss_decode.item()}")
			print(f"Epoch: {epoch+1}, Test Batch Trajectories: {i} Test Total Loss: {total_loss.item()}\n")

			# perform the testing and visualization once every 20 epochs
			if ((epoch)%50 == 0):
				# reshape for all frames in the current trajectory
				for j in range(number_of_trajectories_val):
					cur_trajectory_feature = (torch.reshape(features_OUTPUT_FRAMES_val[:,j,:],(SAMPLING_OUTPUT_FRAMES,number_of_nodes,number_of_features)))*std_val+mean_val
					cur_trajectory_feature_pred = (torch.reshape(y_pred_val[:,j,:],(SAMPLING_OUTPUT_FRAMES,number_of_nodes,number_of_features)))*std_val+mean_val
					# downsample again to 1Hz and select the target node we want (include the very first starting point for the output)
					cur_trajectory_feature = torch.cat((torch.unsqueeze(cur_trajectory_feature[0,target_node,:],0),cur_trajectory_feature[CURRENT_SAMPLING_FREQ-1::CURRENT_SAMPLING_FREQ,target_node,:]),dim=0)
					cur_trajectory_feature_pred = torch.cat((torch.unsqueeze(cur_trajectory_feature_pred[0,target_node,:],0),cur_trajectory_feature_pred[CURRENT_SAMPLING_FREQ-1::CURRENT_SAMPLING_FREQ,target_node,:]),dim=0)

					plt.figure(figsize=(10,10))
					plt.subplot(1, 2, 1).set_xlim([0,40])
					plt.subplot(1, 2, 1).set_title("GT Trajectory")
					plt.plot(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
					plt.scatter(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
					plt.subplot(1, 2, 2).set_xlim([0,40])
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
	print("\n================================================================")

# Create models directory 
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path 
MODEL_NAME = f"model_lstm_batchsize_{BATCH_SIZE}_shuffle_{shuffle}.pkl"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 






















# import random
# # test the model
# # move to main.py after validation
# from data_loader import *
# cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # initialize preliminary information and load data
# torch.manual_seed(42)

# BATCH_SIZE = 1
# shuffle = False
# train_loader,test_loader,train_dataset,test_dataset = load_data(batch_size=BATCH_SIZE,shuffle=shuffle)
# print(f"Length of train loader: {len(train_loader)}")
# print(f"Length of test loader: {len(test_loader)}\n")

# number_of_nodes = train_dataset.num_nodes
# number_of_features = train_dataset.num_node_features
# number_of_out_features = 1024
# stats_train = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
# stats_val = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
# num_layers = 1

# # initialize model and training criterion and optimizer
# model_test = LSTM_EnDecoder(num_nodes=number_of_nodes,num_features=number_of_features,num_out_features=number_of_out_features,n_layer=num_layers,cur_device=cur_device).to(cur_device)

# # Create models directory 
# from pathlib import Path
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
# # Create model save path 
# MODEL_NAME = f"model_lstm_batchsize_{BATCH_SIZE}_shuffle_{shuffle}.pkl"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# check = torch.load(MODEL_SAVE_PATH)
# model_test.load_state_dict(check)
# criterion = torch.nn.MSELoss()  # Define loss criterion.
# train_loss_arr = []
# val_loss_arr = []

# # start the validation phase
# epoch = 0
# val_loss_epoch = 0
# model_test.eval()
# with torch.no_grad():
# 	std_val, mean_val = stats_val[0].to(cur_device),stats_val[1].to(cur_device)
# 	for i, batch_trajectories_val in enumerate(train_loader):
# 		# split into INPUT_FRAMES/OUTPUT_FRAMES 
# 		batch_INPUT_FRAMES_val = batch_trajectories_val[:INPUT_FRAMES]
# 		batch_INPUT_FRAMES_val = batch_INPUT_FRAMES_val[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame
# 		batch_OUTPUT_FRAMES_val = batch_trajectories_val[INPUT_FRAMES:]
# 		batch_OUTPUT_FRAMES_val = batch_OUTPUT_FRAMES_val[DOWNSAMPLE_FREQ-1::DOWNSAMPLE_FREQ] # sample starting from the 10th frame

# 		# get the number of graphs in the current batched trajectory
# 		number_of_trajectories_val = int(batch_trajectories_val[0].num_nodes/number_of_nodes)
		
# 		# normalize and reshape for linear layer
# 		features_INPUT_FRAMES_val,features_OUTPUT_FRAMES_val = model_test.normalize(batch_INPUT_FRAMES_val,batch_OUTPUT_FRAMES_val,number_of_features,
# 									number_of_nodes,number_of_trajectories_val,stats_val)

# 		# perform the forward step
# 		x_pred_val,y_pred_val = model_test(features_INPUT_FRAMES_val,number_of_trajectories_val)
		
# 		# loss function for the current batched trajectories
# 		loss_encode = criterion(x_pred_val, features_INPUT_FRAMES_val)  
# 		loss_decode = criterion(y_pred_val, features_OUTPUT_FRAMES_val)  # Compute the loss
# 		own_loss_encode = torch.sum(torch.pow(torch.abs(x_pred_val-features_INPUT_FRAMES_val),2))/(number_of_features*number_of_trajectories_val*number_of_nodes*SAMPLING_INPUT_FRAMES)
# 		own_loss_decode = torch.sum(torch.pow(torch.abs(y_pred_val-features_OUTPUT_FRAMES_val),2))/(number_of_features*number_of_trajectories_val*number_of_nodes*SAMPLING_OUTPUT_FRAMES)
# 		total_loss = loss_encode + loss_decode
# 		own_total_loss =  own_loss_decode+own_loss_decode
# 		val_loss_epoch += total_loss
# 		print(f"Number of trajectories in Test Batch Trajectories {i}: {number_of_trajectories_val}")
# 		print(f"Epoch: {epoch+1}, Test Batch Trajectories: {i} Test Encoder Loss: {loss_encode.item()}")
# 		print(f"Epoch: {epoch+1}, Test Batch Trajectories: {i} Test Decoder Loss: {loss_decode.item()}")
# 		print(f"Epoch: {epoch+1}, Test Batch Trajectories: {i} Test Total Loss: {total_loss.item()}\n")

# 		# perform the testing and visualization once every 20 epochs
# 		if ((epoch)%50 == 0):
# 			# reshape for all frames in the current trajectory
# 			for j in range(number_of_trajectories_val):
# 				cur_trajectory_feature = (torch.reshape(features_OUTPUT_FRAMES_val[:,j,:],(SAMPLING_OUTPUT_FRAMES,number_of_nodes,number_of_features)))*std_val+mean_val
# 				cur_trajectory_feature_pred = (torch.reshape(y_pred_val[:,j,:],(SAMPLING_OUTPUT_FRAMES,number_of_nodes,number_of_features)))*std_val+mean_val
# 				# downsample again to 1Hz and select the target node we want (include the very first starting point for the output)
# 				cur_trajectory_feature = torch.cat((torch.unsqueeze(cur_trajectory_feature[0,target_node,:],0),cur_trajectory_feature[CURRENT_SAMPLING_FREQ-1::CURRENT_SAMPLING_FREQ,target_node,:]),dim=0)
# 				cur_trajectory_feature_pred = torch.cat((torch.unsqueeze(cur_trajectory_feature_pred[0,target_node,:],0),cur_trajectory_feature_pred[CURRENT_SAMPLING_FREQ-1::CURRENT_SAMPLING_FREQ,target_node,:]),dim=0)

# 				plt.figure(figsize=(10,10))
# 				plt.subplot(1, 2, 1).set_xlim([0,40])
# 				plt.subplot(1, 2, 1).set_title("GT Trajectory")
# 				plt.plot(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
# 				plt.scatter(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
# 				plt.subplot(1, 2, 2).set_xlim([0,40])
# 				plt.subplot(1, 2, 2).set_title("Predicted Trajectory")
# 				plt.plot(cur_trajectory_feature_pred[:,:,0].cpu(),cur_trajectory_feature_pred[:,:,1].cpu())
# 				plt.scatter(cur_trajectory_feature_pred[:,:,0].cpu(),cur_trajectory_feature_pred[:,:,1].cpu())
# 				plt.show()

# 				plt.figure(figsize=(10,10))
# 				plt.xlim(0,40)
# 				plt.plot(cur_trajectory_feature[:,:,0].cpu(),cur_trajectory_feature[:,:,1].cpu())
# 				plt.plot(cur_trajectory_feature_pred[:,:,0].cpu(),cur_trajectory_feature_pred[:,:,1].cpu(),color= "red")
# 				plt.legend(["GT","Pred"])
# 				plt.show()

# 				# get the rmse for each second for some cars
# 				deviation = cur_trajectory_feature-cur_trajectory_feature_pred
# 				deviation_rmse = torch.sqrt(torch.sum(torch.pow(deviation,2),dim=2,keepdim=True))
# 				for k,each_sec in enumerate(deviation_rmse):
# 					print_string = f"Second: {k}"
# 					for h,each_car in enumerate(each_sec):
# 						print_string += f"|| Car {h+1} deviation: {each_car.item():.4f} metres"
# 					print(print_string)
# 		break

# val_loss_arr.append(val_loss_epoch.item()/(i+1))
# print("\n================================================================")
# # %%
