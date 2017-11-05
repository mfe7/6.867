function clusters = filterClusters(clusters,length_threshold_min,length_threshold_max,use_google_map)

%split non-consequtive clusters
new_clusters = struct('id',{},'time',{},'x',{},'y',{},'easting',{},'northing',{},'color',{},'local_x',{},'local_y',{},'cross',{});
remove_ids = [];
for i=1:length(clusters)
    if ~isempty(find(diff(clusters(i).time')>2, 1))
        split_points = [0,find(diff(clusters(i).time')>2), length(clusters(i).time)];
        for s = 2:length(split_points)
            new_clusters(end+1).id = clusters(i).id;
            new_clusters(end).time = clusters(i).time(split_points(s-1)+1:split_points(s));
            new_clusters(end).x = clusters(i).x(split_points(s-1)+1:split_points(s));
            new_clusters(end).y = clusters(i).y(split_points(s-1)+1:split_points(s));
            new_clusters(end).easting = clusters(i).easting(split_points(s-1)+1:split_points(s));
            new_clusters(end).northing = clusters(i).northing(split_points(s-1)+1:split_points(s));
            new_clusters(end).color = clusters(i).color;
            new_clusters(end).vehicle_id = clusters(i).vehicle_id;
        end
        remove_ids = [remove_ids, i];
    end
end
clusters(remove_ids) = [];
clusters = [clusters new_clusters];

% Remove short or long clusters
remove_ids = [];
for i=1:length(clusters)
    if use_google_map
        if or(norm([clusters(i).easting(end),clusters(i).northing(end)]-[clusters(i).easting(1),clusters(i).northing(1)]) < length_threshold_min, norm([clusters(i).easting(end),clusters(i).northing(end)]-[clusters(i).easting(1),clusters(i).northing(1)]) > length_threshold_max)
            remove_ids = [remove_ids, i];
        end
    else
         if or(norm([clusters(i).x(end),clusters(i).y(end)]-[clusters(i).x(1),clusters(i).y(1)]) < length_threshold_min, norm([clusters(i).x(end),clusters(i).y(end)]-[clusters(i).x(1),clusters(i).y(1)]) > length_threshold_max)
            remove_ids = [remove_ids, i];
         end
    end
    if length(clusters(i).time)<2
        remove_ids = [remove_ids, i];
    end
end
clusters(remove_ids) = [];


% Downsample clusters
desired_dt = 0.1;
for i=1:length(clusters)
    current_dt = mode(diff(clusters(i).time));
    downsample_ratio = floor(desired_dt/current_dt);
    if downsample_ratio > 1 && length(clusters(i).time)>downsample_ratio
        time_size = size(clusters(i).time);
        for f =fields(clusters(i))' 
            if isequal(time_size,size(clusters(i).(f{:})))
                clusters(i).(f{:}) = downsample(clusters(i).(f{:}),downsample_ratio);
            end
        end
    end
end