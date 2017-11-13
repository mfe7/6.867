import pdb
from numpy import *
import pylab as pl

# X is data matrix (each row is a data point)
# Y is desired output
# scoreFn is a function of a data point
# values is a list of values to plot
def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))
    zz = array([scoreFn(x) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    
    # Plot contour
    pl.figure()
    pl.scatter
    CS = pl.contour(xx, yy, zz, values, linestyles = 'solid', linewidths = 2)
    pl.clabel(CS)

    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c = squeeze((1.-Y)), s = 50, cmap = pl.cm.cool)
    pl.title(title)
    pl.axis('tight')
    pl.show()