# Training Pipe 
# move to main.py after validation
from data_loader import *
cur_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize preliminary information and load data
torch.manual_seed(42)
BATCH_SIZE = 1
num_of_epochs = 501
shuffle = False
# get the dataloader
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
# Define loss criterion.
criterion = torch.nn.MSELoss() 
# Define optimizer.
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)  
# 
train_loss_arr = []
val_loss_arr = []

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
		
		# normalize input and reshape for model
		features_INPUT_FRAMES,features_OUTPUT_FRAMES = model_0.normalize(batch_INPUT_FRAMES,batch_OUTPUT_FRAMES,number_of_features,
									number_of_nodes,number_of_trajectories,stats_train)
		
		# perform the forward step
		optimizer.zero_grad()  # Clear gradients.
		x_pred,y_pred = model_0(features_INPUT_FRAMES,number_of_trajectories)

		loss_encode = criterion(x_pred, features_INPUT_FRAMES)  
		loss_decode = criterion(y_pred, features_OUTPUT_FRAMES)  # Compute the loss
		own_loss_encode = torch.sum(torch.pow(torch.abs(x_pred-features_INPUT_FRAMES),2))/(number_of_features*number_of_trajectories*number_of_nodes*SAMPLING_INPUT_FRAMES)
		own_loss_decode = torch.sum(torch.pow(torch.abs(y_pred-features_OUTPUT_FRAMES),2))/(number_of_features*number_of_trajectories*number_of_nodes*SAMPLING_OUTPUT_FRAMES)
		total_loss = loss_encode + loss_decode
		train_loss_epoch += total_loss
		total_loss.backward()  # Derive gradients.
		optimizer.step()  # Update parameters based on gradients.

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
			loss_decode = criterion(y_pred_val, features_OUTPUT_FRAMES_val)  # Compute the loss
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
					target_node = 0
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
			break
		

	val_loss_arr.append(val_loss_epoch.item()/(i+1))

	# plot the losses
	if ((epoch+1)%50 == 0):
		plt.figure()
		# plt.ylim(0,1)
		plt.title("Train and Validation Loss per epoch")
		plt.plot(np.arange(0,len(train_loss_arr),1),train_loss_arr)
		plt.plot(np.arange(0,len(train_loss_arr),1),val_loss_arr)
		plt.legend(["train loss","validation loss"])
		plt.show()
	print("================================================================")

# import scipy.io
# pred_np = y_pred_compare_unnormalized.cpu().numpy()
# gt_np = ground_truth_unnormalized.cpu().numpy()
# file_path = 'prediction_visualization\pred.mat'
# file_path2 = 'prediction_visualization\gt.mat'
# scipy.io.savemat(file_path, {'pred_np': pred_np})
# scipy.io.savemat(file_path2, {'gt_np': gt_np})

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

# %%