clear all, clc, close all

paths = ['original-trajectory-data\trajectories-0400-0415-sorted.txt';...
    'original-trajectory-data\trajectories-0500-0515-sorted.txt';...
    'original-trajectory-data\trajectories-0515-0530-sorted.txt'];
scenario = ['0400-0415';'0500-0515';'0515-0530'];
ego_vehicle_id_arr = [3355,2515,2484];

for j = 1:size(paths,1)
    current_path = paths(j,:);
    dataset = readtable(current_path);
    
    % ego vehicle withe most number of frames is ID:2385 with 1168 frames
    ego_vehicle_id = ego_vehicle_id_arr(j);
    
    vehicle_id = dataset.Vehicle_ID;
    frame_id = dataset.Frame_ID;

    % this is the minimum distance to be considered captured in the frame
    % (0.05m)
    min_dist_in_frame = 0.05;
    local_x = dataset.Local_X+min_dist_in_frame;
    local_y = dataset.Local_Y+min_dist_in_frame;
    
    %% find vehicle features within the range of +- 90
    % find frame logical with the ego vehicle IDObtain
    frame_id_with_ego_vehicle_index = (vehicle_id==ego_vehicle_id);
    unique_frame_id_with_ego_vehicle = frame_id(frame_id_with_ego_vehicle_index);
    
    % for each frame, get the vehicles that are within the range of +-90 of the Ego vehicle
    range = 100;
    % figure(1)
    for i = 1:length(unique_frame_id_with_ego_vehicle)
        current_frame_with_ego_vehicle = unique_frame_id_with_ego_vehicle(i);
        
        % get all the vehicles and their features in that frame
        vehicles_in_same_frame_as_ego_index = frame_id==current_frame_with_ego_vehicle;
        vehicle_id_in_same_frame_as_ego = vehicle_id(vehicles_in_same_frame_as_ego_index);
        local_x_of_vehicles_in_same_frame_as_ego = local_x(vehicles_in_same_frame_as_ego_index);
        local_y_of_vehicles_in_same_frame_as_ego = local_y(vehicles_in_same_frame_as_ego_index);
        
        % get ego vehicle's feature for that frame
        index_of_ego_in_current_frame = find(vehicle_id_in_same_frame_as_ego==ego_vehicle_id,1);
        local_x_ego_vehicle = local_x_of_vehicles_in_same_frame_as_ego(index_of_ego_in_current_frame);
        local_y_ego_vehicle = local_y_of_vehicles_in_same_frame_as_ego(index_of_ego_in_current_frame);
        
        % find vehicles that are within the range for that frame
        distance_from_ego = abs(local_y_of_vehicles_in_same_frame_as_ego-local_y_ego_vehicle);
        vehicles_within_ego_range_index = distance_from_ego < range;
        
        % get all the vehicles and their features within the range for that
        % frame
        vehicle_id_within_ego_range = vehicle_id_in_same_frame_as_ego(vehicles_within_ego_range_index);
        local_x_of_vehicles_within_ego_range = local_x_of_vehicles_in_same_frame_as_ego(vehicles_within_ego_range_index);
        local_y_of_vehicles_within_ego_range = local_y_of_vehicles_in_same_frame_as_ego(vehicles_within_ego_range_index);
    
        % visualize the traffic flow around ego 
        plot(local_x_of_vehicles_within_ego_range,local_y_of_vehicles_within_ego_range,'k+',local_x_ego_vehicle,local_y_ego_vehicle,'ro')
        ylim([local_y_ego_vehicle-range local_y_ego_vehicle+range])
        xlim([0 25])
        ylabel("Vertical Distance")
        xlabel("Lane Distance")
        title(sprintf("Plot of vehicles within range of Target Vehicle:%d at frame:%d",ego_vehicle_id,current_frame_with_ego_vehicle))
        text(local_x_of_vehicles_within_ego_range,local_y_of_vehicles_within_ego_range+5,string(vehicle_id_within_ego_range))
        pause()

        % compile trajectory history with ego and its +- 100 scene
        current_frame_array = current_frame_with_ego_vehicle*ones(length(vehicle_id_within_ego_range),1);
        compil_matrix = [vehicle_id_within_ego_range,current_frame_array,local_x_of_vehicles_within_ego_range,local_y_of_vehicles_within_ego_range];
        if(i == 1)
            trajectory_history_ego_scene = [compil_matrix];
        else
            trajectory_history_ego_scene = [trajectory_history_ego_scene;compil_matrix];
        end

        if (i == 400)
            pause()
        end
    end
end

disp("Finished Input Processing")