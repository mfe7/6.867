function [gaussMean, gaussCov, quadBowlA, quadBowlb] = loadParametersP1()

% return the mean, covariance for the gaussian and A, b for the quadratic
% bowl equation specified in the question

data = importdata('parametersp1.txt');

gaussMean = data(1,:)';
gaussCov = data([2,3],:);

quadBowlA = data([4,5],:);
quadBowlb = data(6,:)';


