function ind = clust_time_ind_in_range_finder(x,t)

if isempty(find((x(2:end)-t).*(x(1:end-1)-t)<=0,1))
    ind = [];
else
    [~,ind] = min(abs(x-t));
end