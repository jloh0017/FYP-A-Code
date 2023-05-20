#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import matplotlib.pyplot as plt

INPUT_FRAMES = 40
OUTPUT_FRAMES = 80-INPUT_FRAMES

class LinearLayer(nn.Module):
	def __init__(self, input_frames, output_frames, n_layer=1, bias=True):
		super().__init__()
		self.output_frame = output_frames
		self.n_layer = n_layer

		# layer
		self.linear_relu_stack = nn.Sequential(
					Linear(input_frames, 512),
					nn.ReLU(),
					nn.Linear(512, 1028),
					nn.ReLU(),
					nn.Linear(1028, 512),
					nn.ReLU(),
					nn.Linear(512,output_frames),
				)

	def forward(self, input):
		z = self.linear_relu_stack(input.float())
		return z
	
	def normalize(self,batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,number_of_features,number_of_nodes,number_of_trajectories,stats,cur_device):
		std,mean = stats
		"""
		returns normalized
		features_INPUT_FRAMES: input feature tensor for that current batched trajectory
			Dim: [trajectory #,features,frames]
			trajectory #: details which trajectory in the current batched trajectories
			features: x,y features 
		features_OUTPUT_FRAMES: output feature tensor for that current batched trajectory
			Dim: [trajectory #,features,frames]
			trajectory #: details which trajectory in the current batched trajectories
			features: x,y features 
			frames: details which time frame
		"""

		features_INPUT_FRAMES = []
		for j, frame_batched_trajectories in enumerate(batch_INPUT_FRAMES):
			current_frame_feature = frame_batched_trajectories.x
			normalized_frame_feature = (current_frame_feature-mean)/std
			if j == 0:
				features_INPUT_FRAMES = torch.reshape(normalized_frame_feature,(number_of_trajectories,number_of_features*number_of_nodes,1))
			else:
				test = torch.reshape(normalized_frame_feature,(number_of_trajectories,number_of_features*number_of_nodes,1))
				features_INPUT_FRAMES = torch.cat((features_INPUT_FRAMES,test),dim=2)
		
		features_OUTPUT_FRAMES = []
		for j, frame_batched_trajectories in enumerate(batch_OUTPUT_FRAMES):
			current_frame_feature = frame_batched_trajectories.x
			normalized_frame_feature = (current_frame_feature-mean)/std

			if j == 0:
				features_OUTPUT_FRAMES = torch.reshape(normalized_frame_feature,(number_of_trajectories,number_of_features*number_of_nodes,1))
			else:
				test = torch.reshape(normalized_frame_feature,(number_of_trajectories,number_of_features*number_of_nodes,1))
				features_OUTPUT_FRAMES = torch.cat((features_OUTPUT_FRAMES,test),dim=2)
				
		return features_INPUT_FRAMES.to(cur_device),features_OUTPUT_FRAMES.to(cur_device)
	
# %%

# Training Pipe 
# move to main.py after validation
from data_loader import *
cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize preliminary information and load data
torch.manual_seed(42)

BATCH_SIZE = 2
num_of_epochs = 100
train_loader,test_loader,train_dataset,test_dataset = load_data(batch_size=BATCH_SIZE,shuffle=False)
print(f"Length of train loader: {len(train_loader)}")
print(f"Length of test loader: {len(test_loader)}\n")

number_of_nodes = train_dataset.num_nodes
number_of_features = train_dataset.num_node_features
stats_train = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
stats_test = find_mean_and_std(train_or_val=True,batch_size = BATCH_SIZE)
stats_train = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()] # for checking matrix transformations for unnormalized inputs
stats_val = [torch.Tensor([1,1]).float(),torch.Tensor([0,0]).float()]

# initialize model and training criterion and optimizer
model_0 = LinearLayer(number_of_nodes*number_of_features*INPUT_FRAMES,number_of_nodes*number_of_features*OUTPUT_FRAMES).to(cur_device)
criterion = torch.nn.MSELoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.01)  # Define optimizer.

for epoch in range(num_of_epochs):

	model_0.train()
	for i, batch_trajectories in enumerate(train_loader):
		# split batched trajectories into INPUT_FRAMES/OUTPUT_FRAMES 
		batch_INPUT_FRAMES = batch_trajectories[:INPUT_FRAMES]
		batch_OUTPUT_FRAMES = batch_trajectories[INPUT_FRAMES:]

		# get the number of trajectories in the current batch_trajectories
		number_of_trajectories = int(batch_trajectories[0].num_nodes/number_of_nodes)
		print(f"Number of trajectories in Train Batch Trajectories {i}: {number_of_trajectories}")

		# normalize for linear layer
		features_INPUT_FRAMES,features_OUTPUT_FRAMES = model_0.normalize(batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,number_of_features,
									number_of_nodes,number_of_trajectories,stats_train,cur_device)
		
		# reshape into 1D so that x and y and temporal can interlink
		features_INPUT_FRAMES = torch.reshape(features_INPUT_FRAMES,(number_of_trajectories,number_of_nodes*number_of_features*INPUT_FRAMES))
		features_OUTPUT_FRAMES = torch.reshape(features_OUTPUT_FRAMES,(number_of_trajectories,number_of_nodes*number_of_features*OUTPUT_FRAMES)) 

		# perform the forward step
		optimizer.zero_grad()  # Clear gradients.
		y_pred = model_0(features_INPUT_FRAMES).to(torch.float64)

		loss = criterion(y_pred, features_OUTPUT_FRAMES)  # Compute the loss
		own_loss = torch.sum(torch.pow(torch.abs(y_pred-features_OUTPUT_FRAMES),2))/(number_of_features*number_of_trajectories*number_of_nodes*OUTPUT_FRAMES)
		
		loss.backward()  # Derive gradients.
		optimizer.step()  # Update parameters based on gradients.
		print(f"Epoch: {epoch}, Train Batch Trajectories: {i} Training Loss: {loss.item()}, Self-defined Training Loss: {own_loss}\n")

	model_0.eval()
	with torch.no_grad():
		std_test, mean_test = stats_test[0].to(cur_device),stats_test[1].to(cur_device)
		for i, batch_trajectories_test in enumerate(test_loader):
			# split into INPUT_FRAMES/OUTPUT_FRAMES 
			batch_INPUT_FRAMES_test = batch_trajectories_test[:INPUT_FRAMES]
			batch_OUTPUT_FRAMES_test = batch_trajectories_test[INPUT_FRAMES:]

			# get the number of graphs in the current batched trajectory
			number_of_trajectories_test = int(batch_trajectories_test[0].num_nodes/number_of_nodes)
			print(f"Number of trajectories in Test Batch Trajectories {i}: {number_of_trajectories_test}")

			# normalize and reshape for linear layer
			features_INPUT_FRAMES_test,features_OUTPUT_FRAMES_test = model_0.normalize(batch_INPUT_FRAMES_test,batch_OUTPUT_FRAMES_test,number_of_features,
										number_of_nodes,number_of_trajectories_test,stats_test,cur_device)
			features_INPUT_FRAMES_reshape_test = torch.reshape(features_INPUT_FRAMES_test,(number_of_trajectories_test,number_of_nodes*number_of_features*INPUT_FRAMES))
			features_OUTPUT_FRAMES_reshape_test = torch.reshape(features_OUTPUT_FRAMES_test,(number_of_trajectories_test,number_of_nodes*number_of_features*OUTPUT_FRAMES)) 

			# perform the forward step
			y_pred_test = model_0(features_INPUT_FRAMES_reshape_test).to(torch.float64)
			
			# loss function for the current batched trajectories
			loss_test = criterion(y_pred_test, features_OUTPUT_FRAMES_reshape_test)  # Compute the loss
			own_loss_test = torch.sum(torch.pow(torch.abs(y_pred_test-features_OUTPUT_FRAMES_reshape_test),2))/(number_of_features*number_of_trajectories_test*number_of_nodes*OUTPUT_FRAMES)
			print(f"Epoch: {epoch}, Test Batch Trajectories: {i} Testing Loss: {loss_test.item()}, Self-defined Testing Loss: {own_loss_test}")

			features_OUTPUT_FRAMES_test = torch.reshape(features_OUTPUT_FRAMES_reshape_test,(number_of_trajectories_test,number_of_nodes*number_of_features,OUTPUT_FRAMES))
			y_pred_test = torch.reshape(y_pred_test,(number_of_trajectories_test,number_of_nodes*number_of_features,OUTPUT_FRAMES))

			# visualize the data for each separate trajectories in the batched trajectories
			for j,current_trajectory in enumerate(features_OUTPUT_FRAMES_test):
				ground_truth = torch.reshape(features_OUTPUT_FRAMES_test[j].T,(OUTPUT_FRAMES,number_of_nodes,number_of_features))
				ground_truth_unnormalized = ground_truth*std_test+mean_test

				y_pred_compare = torch.reshape(y_pred_test[j].T,(OUTPUT_FRAMES,number_of_nodes,number_of_features))
				y_pred_compare_unnormalized = y_pred_compare*std_test+mean_test

				total_loss_in_current_trajectory = torch.sum(torch.pow(torch.abs(ground_truth-y_pred_compare),2))/(number_of_nodes*number_of_features*OUTPUT_FRAMES)
				loss = criterion(ground_truth,y_pred_compare)
				print(f" Batch Trajectories: {i}, Trajectory: {j}, Testing Loss: {loss.item()}, Self-defined Loss: {total_loss_in_current_trajectory}")
				print(f"Epoch: {epoch}, Test Batch Trajectories: {i}，Trajectory: {j}, Testing Loss: {loss.item()}, Self-defined Loss: {total_loss_in_current_trajectory}")

				plt.figure()
				plt.xlim(0,90)
				# analyze error in each frame
				for k in range(len(y_pred_compare_unnormalized)):
					gt_current_frame = ground_truth_unnormalized[k]
					y_pred_compare_current_frame = y_pred_compare_unnormalized[k]
					# loss_per_frame_in_current_trajectory = torch.sum(torch.pow(torch.abs(gt_current_frame-y_pred_compare_current_frame),2))/(number_of_nodes*number_of_features)
					# loss = criterion(gt_current_frame,y_pred_compare_current_frame)
					# print(f" batch_trajectories: {i}, Trajectory: {j}, Accuracy: {loss.item()}, Self-defined Accuracy: {loss_per_frame_in_current_trajectory}")

					# check x and y coordinate error
					# for every node
					error = torch.mean(torch.pow(torch.abs(gt_current_frame-y_pred_compare_current_frame),2),dim=0)
					# for one node
					error = torch.pow(torch.abs(gt_current_frame[5,:]-y_pred_compare_current_frame[5,:]),2)
					# print(f" batch_trajectories: {i}, Trajectory: {j}, Frame:{k}, Error_x: {error[0]},Error_y: {error[1]}")
					# print(f" gt_x: {gt_current_frame[5,0]}, gt_y: {gt_current_frame[5,1]}, pred_x:{y_pred_compare_current_frame[5,0]}, pred_y: {y_pred_compare_current_frame[5,1]}")

					# plot for one node in that frame
					plt.scatter(gt_current_frame[5,0].cpu(),gt_current_frame[5,1].cpu(),label= "stars",color= "green")
					plt.scatter(y_pred_compare_current_frame[5,0].cpu(),y_pred_compare_current_frame[5,1].cpu(),label= "stars",color= "red")
				break # test for one trajectory
			plt.show()
			print("\n")
			break # test for one batch trajectories
	print("\n")

import scipy.io
pred_np = y_pred_compare_unnormalized.cpu().numpy()
gt_np = ground_truth_unnormalized.cpu().numpy()
file_path = 'prediction_visualization\pred.mat'
file_path2 = 'prediction_visualization\gt.mat'
scipy.io.savemat(file_path, {'pred_np': pred_np})
scipy.io.savemat(file_path2, {'gt_np': gt_np})

# Create models directory 
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path 
MODEL_NAME = "prediction_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 

# %%
# #############################################################################################################
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
from data_loader import *
cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Create model save path 
MODEL_NAME = "prediction_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Testing pipe
cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize preliminary information and load databatch
torch.manual_seed(42)
