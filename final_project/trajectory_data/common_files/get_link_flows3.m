function observations = get_link_flows3(links, clusters,table_v,color_v,plotting,assumed_walking_speed,radius, angle,x_field,y_field)
try
observations = repmat(struct('times',zeros(2,0),'n',[],'time_made',[],'c',[]),length(links),1);
for i=1:length(observations)
    observations(i).c = {};
end
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
        l(i).c = [];
    end

    clust_time_ind_in_range = cellfun(@(x) clust_time_ind_in_range_finder(x,t), {clusters.time},'UniformOutput',false);
    clust_in_range = ~cellfun(@isempty,clust_time_ind_in_range);

    for c=find(clust_in_range)
        link_num = clusters(c).link_ids(clust_time_ind_in_range{c});
        if ~isempty(find(ismember(links(link_num).points_in_range',clusters(c).link_location(clust_time_ind_in_range{c},:),'rows'),1)) %cluster is projected into observable link region
            l(link_num).n = l(link_num).n+1;
            l(link_num).v = [l(link_num).v clusters(c).velocity];
            l(link_num).c = [l(link_num).c c];
%             observations(link_num).c = [observations(link_num).c c];
            if plotting
    %             h22{c} = plot(links(link_num).points(1,:),links(link_num).points(2,:),'g.');
                h33{c} = plot(clusters(c).(x_field)(clusters(c).time<=t),clusters(c).(y_field)(clusters(c).time<=t),'color',clusters(c).color,'LineWidth',2);
                h55{c} = plot(clusters(c).(x_field)(find(clusters(c).time<=t,1,'last')),clusters(c).(y_field)(find(clusters(c).time<=t,1,'last')),'o','color',clusters(c).color,'LineWidth',2);
            end
        end
    end

    %% For each link that is in range, estimate the flow along that link
    for i=find(cellfun(@(x)size(x,2)>1,{links.points_in_range}))
       if isempty(l(i).v)
            space_mean_speed = assumed_walking_speed;
       else
            space_mean_speed = l(i).n/sum(1./l(i).v);
       end
       dists = sqrt(sum((links(i).points_in_range - repmat(links(i).points(:,1),1,size(links(i).points_in_range,2))).^2));
       times = t-[max(dists);min(dists)]./space_mean_speed;
       observations(i).c{end+1} = l(i).c;
       observations(i).times(:,end+1) = times;
       observations(i).n(end+1) = l(i).n; 
       observations(i).time_made(end+1) = t; 
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
