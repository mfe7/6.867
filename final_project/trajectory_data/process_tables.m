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
date_ = dates(d,:);
display(date_)
table_filename = [table_folder, '/' , 'tables_',num2str(date_(1)),'_',num2str(date_(2)),'_',num2str(date_(3))];
clusters_filename = [clusters_folder, '/' , 'clusters_',num2str(date_(1)),'_',num2str(date_(2)),'_',num2str(date_(3))];clusters_filename = [clusters_folder, '/' , 'clusters_',num2str(date_(1)),'_',num2str(date_(2)),'_',num2str(date_(3))];
clusters2_filename = [clusters_folder, '/' , 'clusters2_',num2str(date_(1)),'_',num2str(date_(2)),'_',num2str(date_(3))];


%% Load the day's data
if exist([table_filename,'.mat'],'file')
    load(table_filename)
end

%% Process the clusters
if do_clusters
if exist([clusters_filename,'.mat'],'file')
    load(clusters_filename)
else
    %% Read clusters from table
    display('Processing clusters');
    clusters = struct('id',{},'time',{},'x',{},'y',{},'easting',{},'northing',{},'color',{});
    clusters2 = struct('id',{},'time',{},'x',{},'y',{},'easting',{},'northing',{},'color',{},'local_x',{},'local_y',{});
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
                t_veh = vehicle_table_v.time;
                t_align = zeros(length(t_veh), length(t_ped));
                for t=1:length(t_ped)
                    t_align(:,t) = t_ped(t) - t_veh;
                end
                t_align = abs(t_align);
                [dt ind_align] = min(t_align);
                
                veh_p1 = [vehicle_table_v.x(ind_align), vehicle_table_v.y(ind_align)];
                ind_align_offset = ind_align + 1;
                offset_complete = 0;
                while ~offset_complete
                    veh_p2 = [vehicle_table_v.x(ind_align_offset), vehicle_table_v.y(ind_align_offset)];
                    veh_delta = veh_p2 - veh_p1;
                    zero_inds = find(~any(veh_delta,2));
                    if length(zero_inds) > 0
                        ind_align_offset(zero_inds) = ind_align_offset(zero_inds) + 1;
                    else
                        offset_complete = 1;
                    end
                end
                veh_delta = veh_p2 - veh_p1;
                r_parallel = normr(veh_delta);
                r_orthog = [-r_parallel(:,2), r_parallel(:,1)];
                
                ped_x = vehicle_table_p.x(vehicle_table_p.ped_id == cluster_ids(i));
                ped_y = vehicle_table_p.y(vehicle_table_p.ped_id == cluster_ids(i));
                d = [ped_x ped_y] - veh_p1;
                ped_parallel = dot(d, r_parallel,2);
                ped_orthog = dot(d, r_orthog,2);
                % Pedestrian position in local vehicle frame
                ped_local = [ped_parallel, ped_orthog];
                
                % Check if ped_local is ever within rectangle in front
                % of vehicle
                top_left = veh_p1 + [10, -2];
                top_right = veh_p1 + [10, 2];
                bottom_left = veh_p1 + [0, -2];
                bottom_right = veh_p1 + [0, 2];
                veh_polygon = [top_left;top_right;bottom_right;bottom_left];
                
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
            
                
                clusters(end+1).id = length(clusters)+1;
                clusters(end).time = vehicle_table_p.time(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).x = vehicle_table_p.x(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).y = vehicle_table_p.y(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).easting = vehicle_table_p.easting(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).northing = vehicle_table_p.northing(vehicle_table_p.ped_id == cluster_ids(i));
                clusters(end).vehicle_id = unique_vehicle_ids{v};
                color = rand(1,2); color(3) = 1-sum(color)/2; color = color(randperm(3)); %Use random darker colors
                clusters(end).color = color;
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
end

%% Process vehicle data
% The vehicle's position data is available in the 'table_v' variable
display('Processing clusters');
if ~isempty(table_v)
    % YOUR CODE HERE: process table_v vehicle data and associate with clusters
    display('table_v:'); display(fieldnames(table_v)); pause(1); %Placeholder
end


end
