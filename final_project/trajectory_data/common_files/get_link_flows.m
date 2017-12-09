function links = get_link_flows(links, clusters,table_v,color_v,plotting,assumed_walking_speed,radius, angle,x_field,y_field)
try
t_old = table_v.time(1);
for ind = 2:size(table_v,1);
%     clc; display('Data playback'); display([num2str(ind),' of ',num2str(size(table_v,1))]);
    t = table_v.time(ind);
    pos = table_v.pos(ind,:);
    if ~isempty(find(strcmp('heading',table_v.Properties.VariableNames),1))
        heading = table_v.heading(ind);
    else
        heading = atan2(diff(table_v.pos(ind-1:ind,2)),diff(table_v.pos(ind-1:ind,1)));
    end
    
    if plotting
        h_v = plotWithCircle(pos,radius,color_v,'d',angle,heading);
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
        l(i).n = 0;
        l(i).v = [];
    end

%     clust_time_ind_in_range = cellfun(@(x) find(x==t,1), {clusters.time},'UniformOutput',false);
    clust_time_ind_in_range = cellfun(@(x) clust_time_ind_in_range_finder(x,t), {clusters.time},'UniformOutput',false);
    clust_in_range = ~cellfun(@isempty,clust_time_ind_in_range);

    for c=find(clust_in_range)
        link_num = find(ismember(reshape([links.connected_nodes],2,[])',clusters(c).connected_nodes(clust_time_ind_in_range{c},:),'rows')');
        l(link_num).n = l(link_num).n+1;
        l(link_num).v = [l(link_num).v clusters(c).velocity];
        if plotting
            h22{c} = plot(links(link_num).points(1,:),links(link_num).points(2,:),'g.');
            h33{c} = plot(clusters(c).(x_field)(clusters(c).time<=t),clusters(c).(y_field)(clusters(c).time<=t),'color',clusters(c).color,'LineWidth',2);
            h55{c} = plot(clusters(c).(x_field)(find(clusters(c).time<=t,1,'last')),clusters(c).(y_field)(find(clusters(c).time<=t,1,'last')),'o','color',clusters(c).color,'LineWidth',2);
        end
    end

    %% For each link that is in range, estimate the flow along that link
    for i=find(cellfun(@(x)size(x,2)>1,{links.points_in_range}))
       if isempty(l(i).v)
            space_mean_speed = assumed_walking_speed;
       else
            space_mean_speed = l(i).n/sum(1./l(i).v);
       end
        observation_length = sqrt(sum(diff(links(i).points_in_range(:,[1,end]),1,2).^2));
        observation_length(observation_length*1.03 > radius) = radius;
%         min_length = space_mean_speed/0.1;
%           min_length = min([links.link_length]);
        min_length = 0;
        if observation_length >= min_length %&& isempty(setdiff([recorded_paths([recorded_paths.time]==t).pedestrian_id],[clusters(clust_in_range).pedestrian_id]))
%             expected_peds = rate(i)*observation_length/pedestrian_vel;
            flow_estimate = l(i).n/observation_length*space_mean_speed;
            links(i).lambda_est.a = links(i).lambda_est.a + flow_estimate;
            links(i).lambda_est.b =  inv(1 + inv(links(i).lambda_est.b));
            links(i).lambda_est_no_prior = [links(i).lambda_est_no_prior flow_estimate];
            links(i).lambda_est_time = [links(i).lambda_est_time t];
%             links(i).link_counts = [links(i).link_counts l(i).n*links(i).link_length/observation_length];
%             links(i).time_counts = [links(i).time_counts links(i).link_length/space_mean_speed];
        end
    end 
    
            %% Update the figure
    if plotting
        if exist('h44','var'), delete(h44); end
        if exist('h22','var'), cellfun(@delete,h22); end
        drawnow; pause(1e-6)
        if exist('h_v','var'), delete(h_v); end
        if exist('h44','var'), delete(h44); end
        if exist('h22','var'), cellfun(@delete,h22); end
        if exist('h55','var'), cellfun(@delete,h55); end
    end

end
catch e
    e.stack.line
end

end

