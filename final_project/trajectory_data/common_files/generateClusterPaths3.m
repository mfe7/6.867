function [clusters,h] = generateClusterPaths3(clusters,links,routes,x_field,y_field,h)

if nargin<6
    plotting=false;
    h=[];
else
    plotting=true;
    cellfun(@delete,h)
end
points = [links.points];
route_links_incidence = [routes.links_incidence];
rm_clusters = [];
for i=1:length(clusters)
    clc; display(['Processing clusters ',num2str(i),' of ',num2str(length(clusters))])
    x = clusters(i).(x_field);
    y = clusters(i).(y_field);
    
    link_ids = zeros(length(x),1);
    link_location = zeros(length(x),2);
    connected_nodes = zeros(length(x),2);
    for j=1:length(x)
        if j==1
            point_heading = atan2(diff(y(1:2)),diff(x(1:2)));
        else
            point_heading = atan2(diff(y(j-1:j)),diff(x(j-1:j)));
        end
        distance_from_known_points = sqrt(sum((points - repmat([x(j);y(j)],1,length(points))).^2));
        b = find(distance_from_known_points<=3);
        if isempty(b)
            [a,b] = sort(distance_from_known_points);
            b = b(1:2);
            if a(1)>20
                rm_clusters = [rm_clusters i];
                break;
            end;
        end
        best_links = unique(floor((b-1)/(size(points,2)/length(links)))+1);
        heading_diff = point_heading-[links(best_links).heading];
        heading_diff(heading_diff > pi) = heading_diff(heading_diff > pi) - 2*pi;
        heading_diff(heading_diff < -pi) = 2*pi + heading_diff(heading_diff < -pi);
%         [~,best_link_ind] = min(abs(heading_diff));
        best_link_ind = find(abs(abs(heading_diff) - min(abs(heading_diff))) < 0.01,1);
        link_ids(j) = best_links(best_link_ind);
%         link_location(j,:) = points(:,b(best_link_ind))';
%         connected_nodes(j,:) = links(link_ids(j)).connected_nodes;
    end
    
    if ~isempty(find(ismember(rm_clusters,i), 1)), continue; end
    
    unique_link_ids = unique(link_ids);
    if length(unique_link_ids) > 1
        counts = hist(link_ids,unique_link_ids);
        [~,sorted_id] = sort(counts,2, 'ascend');
        unique_link_ids = unique_link_ids(sorted_id);
        for j=1:length(unique_link_ids)
            id_list = unique_link_ids(j:end);
            if any(all(route_links_incidence(id_list,:)));
                break;
            end
        end
    else
        id_list = unique_link_ids;
    end
    
    allowable_points = [links(id_list).points];
    for j=1:length(x)
        distance_from_known_points = sqrt(sum((allowable_points - repmat([x(j);y(j)],1,length(allowable_points))).^2));
        [~,b] = min(distance_from_known_points);
        best_link_ind = floor((b-1)/(size(allowable_points,2)/length(id_list)))+1;
        link_ids(j) = id_list(best_link_ind);
        link_location(j,:) = allowable_points(:,b)';
        connected_nodes(j,:) = links(link_ids(j)).connected_nodes;
    end
    
    clusters(i).connected_nodes = connected_nodes;
    clusters(i).link_ids = link_ids;
    clusters(i).link_location = link_location;
    
end

clusters(rm_clusters) = [];