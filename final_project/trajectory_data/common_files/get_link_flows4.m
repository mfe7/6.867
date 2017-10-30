function observations = get_link_flows4(links, clusters,table_v,assumed_walking_speed,radius, angle, plotting)

if nargin<7
    plotting = 0;
end

desired_interval_length = 10;

try
observations = repmat(struct('times',zeros(2,0)),length(links),1);

for ind = 2:size(table_v,1);
    clc; display('Data playback'); display([num2str(ind),' of ',num2str(size(table_v,1))]);
    t = table_v.time(ind);
    pos = table_v.pos(ind,:);
    if ~isempty(find(strcmp('heading',table_v.Properties.VariableNames),1))
        heading = table_v.heading(ind);
    else
        heading = atan2(diff(table_v.pos(ind-1:ind,2)),diff(table_v.pos(ind-1:ind,1)));
    end
    
    if plotting
        h_v = plotWithCircle(pos,radius,'red','d',angle,heading);
    end
   
    for i=1:length(links)
        points = [links(i).points];
        points_vec = (points - repmat(pos,size(points,2),1)');
        distance_away = sqrt(sum(points_vec.^2));
        heading_diff = heading - atan2(points_vec(2,:),points_vec(1,:));
        heading_diff(heading_diff > pi) = heading_diff(heading_diff > pi) - 2*pi;
        heading_diff(heading_diff < -pi) = 2*pi + heading_diff(heading_diff < -pi);
        in_range = and(distance_away <= radius , abs(heading_diff) <= angle/2);
        links(i).points_in_range = points(:,in_range);
        if plotting && ~isempty(links(i).points_in_range)
            h44(i) = plot(links(i).points_in_range(1,:),links(i).points_in_range(2,:),'kd');
        end
    end

    %% For each link that is in range, estimate the observation times on that link
    for i=find(cellfun(@(x)size(x,2)>1,{links.points_in_range}))
       space_mean_speed = links(i).assumed_walking_speed;
       dists = sqrt(sum((links(i).points_in_range - repmat(links(i).points(:,1),1,size(links(i).points_in_range,2))).^2));
       times = t-[max(dists);min(dists)]./space_mean_speed;
       observations(i).times(:,end+1) = times;
    end     
    
    %% For each link that is in range, estimate the flow along that link
%     for i=find(cellfun(@(x)size(x,2)>1,{links.points_in_range}))
%        space_mean_speed = links(i).assumed_walking_speed;
%        dists = sqrt(sum((links(i).points_in_range - repmat(links(i).points(:,1),1,size(links(i).points_in_range,2))).^2));
%        times = t-[max(dists);min(dists)]./space_mean_speed;
%        overlaps = ~or(min(times)>max(observations(i).times,[],1),max(times)<min(observations(i).times,[],1));
% %        if isempty(overlaps)
%            f_overlaps = find(overlaps);
%            for o=f_overlaps
%                if observations(i).times(1,o) < times(1)
%                     observations(i).times(2,o) = times(1)-1e-4;
%                else
%                     observations(i).times(1,o) = times(2)+1e-4;
%                end
% 
%            end
% %        end
%        observations(i).times(:,end+1) = times;
%     end 
%     
     %% Update the figure
    if plotting
        drawnow; pause(1e-6)
        if exist('h_v','var'), delete(h_v); end
        if exist('h44','var'), delete(h44); end
    end
end

% Merge overalapping observation regions
for i=1:length(observations)
    A = observations(i).times';
    n = size(A,1);
    [t,p] = sort(A(:));
    z = cumsum(accumarray((1:2*n)',2*(p<=n)-1));
    z1 = [0;z(1:end-1)];
    A2 = [t(z1==0 & z>0),t(z1>0 & z==0)];
    A3 = [];
    for j=1:size(A2,1)
        num_new_points = max(diff(A2(j,1:end))/desired_interval_length,2);
        interval_limits = linspace(A2(j,1),A2(j,end), num_new_points);
        A3 = [A3 ; [interval_limits(1:end-1)' , interval_limits(2:end)']];
    end
    observations(i).times = A3';
    observations(i).cluster_ids = cell(1,size(observations(i).times,2));
    observations(i).n = zeros(1,size(observations(i).times,2));
end

% Assign clusters to observation regions
for i=1:length(clusters)
    for j=1:length(clusters(i).unique_link_ids)
        link_id = clusters(i).unique_link_ids(j);
        start_time = clusters(i).start_time_estimates(j);
        time_interval = and(min(observations(link_id).times,[],1) <= start_time , max(observations(link_id).times,[],1) >= start_time);
        if isempty(observations(link_id).times)
            continue;
        elseif ~isempty(find(time_interval, 1))
            observations(link_id).cluster_ids{find(time_interval,1)} = [clusters(i).id , observations(link_id).cluster_ids{find(time_interval,1)}]; %#ok<*FNDSB>
            observations(link_id).n(find(time_interval,1)) = observations(link_id).n(find(time_interval,1)) + 1;
        else
            [~,best_time_interval] = min(abs(observations(link_id).times-start_time),[],2);
            observations(link_id).cluster_ids{best_time_interval(1)} = [clusters(i).id , observations(link_id).cluster_ids{best_time_interval(1)}];
            observations(link_id).n(best_time_interval(1)) = observations(link_id).n(best_time_interval(1)) + 1;
        end
    end
end


catch e
    e.stack.line
    rethrow(e)
end

end



%         time_interval = and(min(observations(link_id).times,[],1) < start_time , max(observations(link_id).times,[],1) >= start_time);
%         if isempty(find(time_interval, 1))
%             if isempty(observations(link_id).times)
%                 observations(link_id).times = [start_time-tack_on; start_time+tack_on];
%                 observations(link_id).cluster_ids = {clusters(i).id};
%                 observations(link_id).n = [1];
%                 continue;
%             end
%             next_interval_ind = find(observations(link_id).times(1,:)-start_time > 0,1);
%             if isempty(next_interval_ind)
%                 next_interval = [inf, inf];
%                 previous_interval_ind = length(observations(link_id).times(1,:));
%             else
%                 next_interval = observations(link_id).times(:,next_interval_ind);
%                 previous_interval_ind = next_interval_ind-1;
%             end
%             if previous_interval_ind < 1
%                 previous_interval = [-inf, -inf];
%             else
%                 previous_interval = observations(link_id).times(:,previous_interval_ind);
%             end
%             new_interval = [max(start_time-tack_on, previous_interval(2)) ; min(start_time+tack_on, next_interval(1))];
%             observations(link_id).times = [observations(link_id).times(:,1:previous_interval_ind) , new_interval , observations(link_id).times(:,next_interval_ind:end)];
%             observations(link_id).cluster_ids = {observations(link_id).cluster_ids{1:previous_interval_ind} , clusters(i).id , observations(link_id).cluster_ids{next_interval_ind:end} };
%             observations(link_id).n = [observations(link_id).n(1:previous_interval_ind), 1 , observations(link_id).n(next_interval_ind:end)];
%         else
%             observations(link_id).cluster_ids{find(time_interval)} = [clusters(i).id , observations(link_id).cluster_ids{find(time_interval)}]; %#ok<*FNDSB>
%             observations(link_id).n(find(time_interval)) = observations(link_id).n(find(time_interval)) + 1;
%         end

