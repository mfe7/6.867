clear; close all; clc;

addpath(genpath('.'))

celloutputs = @(x) x{:};
num2struct = @(x) celloutputs(num2cell(x,1));
num2struct2 = @(x) celloutputs(num2cell(x,2));

%% Parameters
do_clusters = true;

load('april')
network_name = 'stata_kendall_green';
use_google_map = 1;
length_threshold_min = 0;
length_threshold_max = inf;
assumed_walking_speed = 1.5;
radius = 20;
angle = 160*pi/180;
plotting = 0;
static = '';

dates = datevec(datenum([2016, 2, 1]):datenum([2016, 6, 30]));
days = 1:size(dates,1);

%folders
table_folder = ['trajectory_tables',static];
clusters_folder = ['clusters_',network_name,static];
if ~exist(['./',table_folder],'dir'), mkdir(['./',table_folder]); end
if ~exist(['./',clusters_folder],'dir'), mkdir(['./',clusters_folder]); end

%% Load map
if plotting
    figure('name','data'); 
    if use_google_map
        imshow(flipud(street_map.image),street_map.ref)
    else
        imshow(flipud(grid_map.image),grid_map.ref)
    end
    set(gca,'ydir','normal')
    axis off
    hold on;
    pause(1e-9);
end

%% Load graph
[nodes,routes,odpairs,links] = getGraphInfo(network_name,0,plotting);

%% Process data one day at a time
for d = days;
clear table_v table_p clusters
%% Setup file data    
date_ = dates(days(d),:);
display(date_)
table_filename = [table_folder, '/' , 'tables_',num2str(date_(1)),'_',num2str(date_(2)),'_',num2str(date_(3))];
clusters_filename = [clusters_folder, '/' , 'clusters2_',num2str(date_(1)),'_',num2str(date_(2)),'_',num2str(date_(3))];clusters_filename = [clusters_folder, '/' , 'clusters_',num2str(date_(1)),'_',num2str(date_(2)),'_',num2str(date_(3))];


%% Load the day's data
if exist([table_filename,'.mat'],'file')
    load(table_filename)
end

%% Process vehicle data
% The vehicle's position data is available in the 'table_v' variable
display('Processing vehicle data...');
t_jump = 0.5; % upper bound on typical time between successive data points
pos_jump = 1.0; % upper bound on typical distance between successive data points
t_long_enough = 5.0; % lower bound on time duration of a useful vehicle trajectory segment
if ~isempty(table_v)
    unique_vehicle_ids = unique(table_p.vehicle_id);
    for v=1:length(unique_vehicle_ids) % go through each vehicle
        vehicle_table_v = table_v(strcmp(table_v.vehicle_id,unique_vehicle_ids(v)),:);
        
        % Extract time, position from vehicle table into matrices
        t = vehicle_table_v{:,{'time'}};
        pos = vehicle_table_v{:,{'x','y'}};
        
        % Find indices where timestamp/position jumps
        % 1a) Find difference between consecutive timestamps
        dt = diff(t); 
        % 1b) Find distance between consecutive vehicle data pts
        dpos = diff(pos);
        ddist = sqrt(sum(dpos.^2,2));
        % 2) Find indices corresponding to jumps
        % note: bad_inds is a list of indices where [..., bad_inds] and
        % [bad_inds+1,...] should be separated. That is, bad_inds to
        % bad_inds+1 is where the jump occurs.
        bad_dt_inds = find(dt > t_jump);
        bad_dpos_inds = find(ddist > pos_jump);
        bad_inds = union(bad_dt_inds, bad_dpos_inds);
        
        % Find start/end times of trajectory segments between
        % timestamp/position jumps and put into valid_t:
        % valid_t =
        %   [t_start, t_end, duration (sec), lower index, upper index]
        %       (one row per smooth trajectory segment)
        % 1) Initialize from t=t0 to first jump
        valid_t = [];
        t_start = vehicle_table_v{1,{'time'}};
        t_end = vehicle_table_v{bad_inds(2),{'time'}};
        duration = t_end - t_start;
        if duration > t_long_enough
            valid_t = [valid_t; t_start, t_end, duration, 1, bad_inds(2)];
        end
        % 2) Iterate through the rest of the jumps and add to valid_t
        for i=2:(length(bad_inds)-1)
            t_start = vehicle_table_v{bad_inds(i)+1,{'time'}};
            t_end = vehicle_table_v{bad_inds(i+1),{'time'}};
            duration = t_end - t_start;
            if duration > t_long_enough
                valid_t = [valid_t; t_start, t_end, duration, bad_inds(i)+1, bad_inds(i+1)];
            end
        end
        
        % smooth_veh_traj =
        %   [timestamp, global_x, global_y, heading (rad), r_par, r_orthog]
        %       where r_par, r_orthog is the unit vector pointing out of the
        %       vehicle's front
        %       (one row per timestamp)
        smooth_veh_traj = find_smooth_veh_traj(vehicle_table_v, valid_t);

    end


end
display('Done processing vehicle trajectory');

%% Process the clusters
% if do_clusters
if 0
% if exist([clusters_filename,'.mat'],'file')
    load(clusters_filename)
else
    %% Read clusters from table
    display('Processing clusters');
    clusters = struct('id',{},'time',{},'x',{},'y',{},'easting',{},'northing',{},'color',{},'local_x',{},'local_y',{},'cross',{});
    if ~isempty(table_p)
        
        num_crosses = 0;
        unique_vehicle_ids = unique(table_p.vehicle_id);
        for v=1:length(unique_vehicle_ids) % go through each vehicle
            vehicle_table_p = table_p(strcmp(table_p.vehicle_id,unique_vehicle_ids(v)),:);
            vehicle_table_v = table_v(strcmp(table_v.vehicle_id,unique_vehicle_ids(v)),:);
            cluster_ids = unique(vehicle_table_p.ped_id);
            for i=1:length(cluster_ids) % go through each cluster
                
                % Extract pedestrian cluster time vector, and vehicle's
                % time vector.
                % Find index of vehicle's time vector that's closest to
                % each pedestrian timestamp to make sure positions
                % are synchronized.
                t_ped = vehicle_table_p.time(vehicle_table_p.ped_id == cluster_ids(i));
                
                % Check that t_ped's upper/lower bounds are within a 
                % vehicle trajectory window. Skip ped cluster if it
                % occurs during a vehicle pos/time jump.
                tmp = valid_t(:,1) - t_ped(1);
                ix = find(tmp>0,1);
                if t_ped(end) > valid_t(ix,2)
                    continue;
                end
                
                t_veh = smooth_veh_traj(:,1);
                t_align = zeros(length(t_veh), length(t_ped));
                for t=1:length(t_ped)
                    t_align(:,t) = t_ped(t) - t_veh;
                end
                t_align = abs(t_align);
                [dt ind_align] = min(t_align);
                
                veh_pos = [smooth_veh_traj(ind_align,2:3)];
                r_parallel = smooth_veh_traj(ind_align,5:6);
                r_orthog = [-r_parallel(:,2), r_parallel(:,1)];
                
                ped_x = vehicle_table_p.x(vehicle_table_p.ped_id == cluster_ids(i));
                ped_y = vehicle_table_p.y(vehicle_table_p.ped_id == cluster_ids(i));
                d = [ped_x ped_y] - veh_pos;
                ped_parallel = dot(d, r_parallel,2);
                ped_orthog = dot(d, r_orthog,2);
                % Pedestrian position in local vehicle frame
                ped_local = [ped_orthog, ped_parallel];
                
                % Check if ped_local is ever within rectangle in front
                % of vehicle
                veh_polygon = [];
                top_left = [-2, 10];
                top_right = [2, 10];
                bottom_left = [-2, 0];
                bottom_right = [2, 0];
                veh_polygon = [top_left; top_right; bottom_right; bottom_left];

                ped_crosses_in_front = 0;
                for t=1:length(t_ped)
                    if inpolygon(ped_local(t,1), ped_local(t,2), veh_polygon(:,1), veh_polygon(:,2))
                        ped_crosses_in_front = 1;
                    end
                end
                if ped_crosses_in_front
                    num_crosses = num_crosses + 1;
                    display('cross!')
                else
                    display('no cross.')
                end
                
                % Plot global and local frames for a single cluster
                if plotting == 1
                    clf;
                    subplot(1,2,1);
                    hold on;
                    veh_pos_plot = plot(veh_pos(:,1), veh_pos(:,2),'r--o');
                    veh_start_plot = plot(veh_pos(1,1), veh_pos(1,2),'r*');
                    veh_end_plot = plot(veh_pos(end,1), veh_pos(end,2),'rx');
                    ped_pos_plot = plot(ped_x, ped_y,'b--o');
                    ped_start_plot = plot(ped_x(1), ped_y(1),'b*');
                    ped_end_plot = plot(ped_x(end), ped_y(end),'bx');
                    legend([veh_pos_plot, ped_pos_plot],{'Vehicle','Pedestrian'});
                    xlabel('x (m)');
                    ylabel('y (m)');
                    title('Global Frame');
                    subplot(1,2,2);
                    hold on;
                    plot(ped_local(:,1), ped_local(:,2),'b--o');
                    plot(ped_local(1,1), ped_local(1,2),'b*');
                    plot(ped_local(end,1), ped_local(end,2),'bx');
                    rectangle('Position',[-1 -3 2 3],'EdgeColor','blue');
                    rectangle('Position',[-2 0 4 10],'LineStyle','--','EdgeColor','red');
                    if ped_crosses_in_front
                       text(-1,-0.5,'CROSS!'); 
                    end
                    title('Vehicle`s Local Frame');
                    xlabel('x (m)');
                    ylabel('y (m)');
                    pause(0.2);
                end
                
                clusters(end+1).id = length(clusters)+1;
                clusters(end).time = vehicle_table_p.time(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).x = vehicle_table_p.x(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).y = vehicle_table_p.y(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).easting = vehicle_table_p.easting(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).northing = vehicle_table_p.northing(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).vehicle_id = unique_vehicle_ids{v};
                color = rand(1,2); color(3) = 1-sum(color)/2; color = color(randperm(3)); %Use random darker colors
                clusters(end).color = color;
                clusters(end).local_x = ped_local(:,1);
                clusters(end).local_y = ped_local(:,2);
                clusters(end).cross = ped_crosses_in_front;
            end
        end
        display(num_crosses)
        display(length(cluster_ids))
        display(num_crosses/length(cluster_ids))
        display("done")

        %% Remove short/long clusters
        clusters = filterClusters(clusters,length_threshold_min,length_threshold_max,use_google_map);

        % Get path information
        clusters = generateClusterPaths3(clusters,links,routes,'easting','northing');

        % Get cluster velocities
        [clusters.velocity] = num2struct(arrayfun(@(cluster) mean(sqrt(sum(diff([cluster.easting;cluster.northing]').^2,2))./diff(cluster.time)'), clusters));
        
        clusters = merge_and_estimate_cluster_arrivals(clusters,links);
    end
    save(clusters_filename,'clusters')
end
display('Done processing clusters.')
end


