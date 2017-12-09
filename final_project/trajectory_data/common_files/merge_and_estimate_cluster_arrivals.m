function clusters = merge_and_estimate_cluster_arrivals(clusters,links)

cutoff = 5;

normfun = @(A) sqrt(sum(A.^2,2));

%% compute distance from node start
for i=1:length(clusters)
    clusters(i).unique_link_ids = unique(clusters(i).link_ids)';
    for j= 1:length(clusters(i).unique_link_ids)
        valid_inds = clusters(i).link_ids==clusters(i).unique_link_ids(j);
        start_point = links(clusters(i).unique_link_ids(j)).points(:,1);
        current_points = clusters(i).link_location(valid_inds,:);
        clusters(i).distance_away(valid_inds,:) = normfun(current_points - repmat(start_point',size(current_points,1),1));
    end
end

%% Merge clusters that match a predicted siteing
cluster_match_distances = inf*ones(length(clusters));
for i=1:length(clusters)
    for j=1:length(clusters)
        same_links = clusters(i).link_ids(end) == clusters(j).link_ids(1);
        min_val_overlaps = and(min(clusters(i).time)>=min(clusters(j).time),min(clusters(i).time)<=max(clusters(j).time));
        max_val_overlaps = and(max(clusters(i).time)>=min(clusters(j).time),max(clusters(i).time)<=max(clusters(j).time));
        not_over_lapping_times = ~or(min_val_overlaps,max_val_overlaps);
        similar_velocities = abs(clusters(i).velocity - clusters(j).velocity) < 0.1;
        different_vehicles = ~strcmp(clusters(i).vehicle_id,clusters(j).vehicle_id);
        if same_links && similar_velocities && (not_over_lapping_times || different_vehicles)
            valid_inds = clusters(j).link_ids==clusters(j).link_ids(1);
            time_act = clusters(j).time(valid_inds');
            da_act = clusters(j).distance_away(valid_inds');
            time_pred = time_act - clusters(i).time(end);
            da_pred = clusters(i).distance_away(end,:) + mean([clusters(i).velocity, clusters(j).velocity]) * time_pred;
            if ~isequal(size(da_pred),size(da_act)), da_pred = da_pred';end
            cluster_match_distances(i,j) = min(abs(da_pred-da_act));
        end
    end
end

% pid_cell = {clusters.pedestrian_id};
% [clusters.pedestrian_ids] = pid_cell{:};
previously_matched_clusters = zeros(1,length(clusters));
for j=1:length(clusters)
    matching_cluster_ids = find(cluster_match_distances(:,j)<cutoff);
    [~,b] = min(cluster_match_distances(matching_cluster_ids,j));
    if ~isempty(b)
        i = matching_cluster_ids(b);
        [~,corresponding_best_match] = min(cluster_match_distances(:,i));
        if corresponding_best_match ~= j
            continue;
        end
        if previously_matched_clusters(i)>0, 
            i = previously_matched_clusters(i);
            if i==j
                continue;
            end
        end
        if size(clusters(i).time,1)>size(clusters(i).time,2)
            clusters(i).time = [clusters(i).time; clusters(j).time];
            clusters(i).x = [clusters(i).x; clusters(j).x];
            clusters(i).y = [clusters(i).y; clusters(j).y];
            clusters(i).easting = [clusters(i).easting; clusters(j).easting];
            clusters(i).northing = [clusters(i).northing; clusters(j).northing];
        else
            clusters(i).time = [clusters(i).time clusters(j).time];
            clusters(i).x = [clusters(i).x clusters(j).x];
            clusters(i).y = [clusters(i).y clusters(j).y];
            clusters(i).easting = [clusters(i).easting clusters(j).easting];
            clusters(i).northing = [clusters(i).northing clusters(j).northing];
        end
        clusters(i).connected_nodes = [clusters(i).connected_nodes; clusters(j).connected_nodes];
        clusters(i).link_ids = [clusters(i).link_ids; clusters(j).link_ids];
        clusters(i).link_location = [clusters(i).link_location; clusters(j).link_location];
        clusters(i).distance_away = [clusters(i).distance_away; clusters(j).distance_away];
        clusters(i).velocity = mean([clusters(i).velocity, clusters(j).velocity]);
        clusters(i).vehicle_id = unique([clusters(i).vehicle_id, clusters(j).vehicle_id]);
%         clusters(i).pedestrian_ids = [clusters(i).pedestrian_ids clusters(j).pedestrian_ids];
        previously_matched_clusters(j) = i;
    end
end
clusters(previously_matched_clusters>0) = []; 

%% estimate cluster start_time
for i=1:length(clusters)
    for j= 1:length(clusters(i).unique_link_ids)
        valid_inds = clusters(i).link_ids==clusters(i).unique_link_ids(j);
        if size(clusters(i).time,1)>size(clusters(i).time,2)
            start_time_estimate = clusters(i).time(valid_inds) - clusters(i).distance_away(valid_inds)/clusters(i).velocity;
        else
            start_time_estimate = clusters(i).time(valid_inds)' - clusters(i).distance_away(valid_inds)/clusters(i).velocity;
        end
        clusters(i).start_time_estimates(j) = mean(start_time_estimate);
%         ind = pedestrians(clusters(i).pedestrian_id).link_and_start_time(1,:) == clusters(i).unique_link_ids(j);
%         error(i,j) = pedestrians(clusters(i).pedestrian_id).link_and_start_time(2,ind) - clusters(i).start_time_estimates(j);
    end
end
    