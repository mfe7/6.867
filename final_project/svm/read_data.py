import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from collections import OrderedDict # ordered import of keys to dict
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.patches as patches
import os 

##############################################
# Use latex font in matplotlib plots
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
##############################################

dir_path = os.path.dirname(os.path.realpath(__file__))

class Data:
  def __init__(self, file_name = 'clusters_2016_2_9'):
    self.file_name = file_name
    self._data = {}
    self._t_steps = 10 # length of trajectory sliplets
    self._X = np.zeros((0, 2 * self._t_steps))
    self._Y = np.zeros((0, 1))
    self._d_t = 0
    self._init_data()

  def _init_data(self):
    cluster_keys = ['id', 'time', 'x', 'y', 'easting', 'northing', 'color', 'local_x', 'local_y', 'cross', 'vehicle_id']
    self._data = OrderedDict.fromkeys(cluster_keys)

    # TODO: Run read_data() over file_name array and concatenate data 
    self.read_data()
    self.set_XY()

  def read_data(self):
    path = '../trajectory_data/clusters_stata_kendall_green/'+ self.file_name +'.mat'
    data_full_mat = sio.loadmat(path) #['clusters2', '__version__', '__header__', '__globals__']

    self.data_full = data_full_mat['clusters'] #(1,906)
    #self.ntraj = self.data_full.shape[1] 

  # return an [x1x2 x1x2 ...] X vector, ordered by t, with length 1/delta_t
  # splices the full vector into ntraj*nsnips elements of length _t_steps
  # the beginning of one snip is after the end of the previous 
  def set_XY(self):
    x1_id = self._data.keys().index('local_x')
    x2_id = self._data.keys().index('local_y')
    y_id = self._data.keys().index('cross')

    # find dimensions
    ntraj = self.data_full.shape[1]
    
    x_tmp = np.zeros((1, 2*self._t_steps))

    for traj_id in range(ntraj):
      traj_len = self.data_full[0][traj_id][x1_id].shape[0]
      nsnips = traj_len // self._t_steps # cuts off the remainder trajectory < _t_steps
      
      for snip_id in range(nsnips):
        snip_range = range(snip_id*self._t_steps,((snip_id+1)*self._t_steps))
        
        # get snips
        x1 = self.data_full[0][traj_id][x1_id][snip_range]
        x2 = self.data_full[0][traj_id][x2_id][snip_range]
        y = self.data_full[0][traj_id][y_id]

        # alternate x1x2
        x_tmp[0, ::2] = np.reshape(x1[:], x1.shape[0])
        x_tmp[0, 1::2] = np.reshape(x2[:], x2.shape[0])        
        
        self._X = np.append(self._X, x_tmp, axis = 0)
        self._Y = np.append(self._Y, y, axis = 0)

    nsnips_total = self._Y.shape[0]
    print('[STATUS] compiled {} to {} trajectories with t stepsize {}'.format(ntraj, nsnips_total, self._t_steps))

  def get_XY(self):
    return self._X, np.reshape(self._Y, self._Y.shape[0])

  # saves XY vector to csv file to reduce loading time
  def save_to_csv():
    return 0

  # Read in first data point
  def set_data(self):
    for i, key in enumerate(self._data.keys()):
      self._data[key] = self.data_full[0][0][i]
    
  def get_data(self):
    return self._data

  def plot_clf(self, x, y, max_npoints):
    x1 = x[:, ::2]
    x2 = x[:, 1::2]

    fig, ax = plt.subplots()
    # Plot trained curve on new data
    print("[INFO] Dimensions of spliced trajectory set (ntraj x length): {} x {}".format(x1.shape[0], x1.shape[1]))
    for i in range(x1.shape[0]):
      if i < max_npoints:
        if y[i] == 0:
          ax.plot(x1[i], x2[i], '-', color = 'green')
        else:
          ax.plot(x1[i], x2[i], '-', color = 'red')#, label = 'y != 1') 


    # Add image of vehicle at (0,0)
    fn = get_sample_data(dir_path+"/car_from_above.png", asfileobj=False)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.2)
    imagebox.image.axes = ax
    xy = [0,-3.5]
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0., 0.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        )
    ax.add_artist(ab)

    # Add rectangle to depict "forbidden zone"
    forbidden_zone_width = 4
    forbidden_zone_height = 10
    rectangle_bottom_left = (0-forbidden_zone_width/2.0, 0)
    rect = patches.Rectangle(rectangle_bottom_left,forbidden_zone_width,forbidden_zone_height,linewidth=10,edgecolor='b',facecolor='none',linestyle='--')
    ax.add_patch(rect)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 30)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('red = cross, green = no cross')
    plt.show()


  

