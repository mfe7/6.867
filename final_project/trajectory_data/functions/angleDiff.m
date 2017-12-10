function [ angle_diff ] = angleDiff( angle_1, angle_2 )
%ANGLEDIFF subtract two angles and wrap around [-pi,pi]
% angle_1, angle_2 are in radians between [-pi, pi]
    angle_diff_raw = angle_1 - angle_2;
	angle_diff = mod(angle_diff_raw + pi, 2 * pi) - pi;
end

