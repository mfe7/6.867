####################
# Notes from Justin
####################

The data collected by the vehicles is stored in "trajectory table" .mat files for each data. Each file contains a table_p and a table_v saved table variable containing pedestrian and vehicle data, respectively. If no data was collected for a day, the tables will be empty. 

'process_tables.m': will parse through the data for each day, extract clusters from table_p, and save the process data in a directory. To extract vehicle data, simply add your code to the "YOUR CODE HERE" section where you can access the table_v data for vehicle trajectory data.

'trajectory_tables_viewer.m': will display the cluster data on a campus map


######################
# 6.867 Final Project
# (updated 11/19/17)
######################

In order to compute/extract clusters in the vehicle's local frame, use the file process_tables.m
and .mat files will be output into the clusters_<location> directory.