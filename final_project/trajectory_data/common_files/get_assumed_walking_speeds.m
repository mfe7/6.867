function links = get_assumed_walking_speeds(clusters,links)

cluster_link_incidence = cell2mat(cellfun(@ismember,num2cell(repmat([1:length(links)],length(clusters),1),2),{clusters.unique_link_ids}','UniformOutput',false));
link_velocities = sum(cluster_link_incidence)./((1./[clusters.velocity])*cluster_link_incidence); % space mean speed
link_velocities(isnan(link_velocities)) = mean(link_velocities(~isnan(link_velocities)));
link_velocities(isnan(link_velocities)) = 1.5; %if everthing is nan, just use 1.5
link_velocities_cell = num2cell(link_velocities);
[links.assumed_walking_speed] = link_velocities_cell{:};