import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.random import randn
import csv
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import os.path

def read_2D(fname) :
  nox    = 0
  noy    = 0
  lineNo = 0
  rownum = 0
  X=[]
  Y=[]
  Z=[]
  with open(fname, 'rb') as f:
    reader = csv.reader(f)
    for row in reader :
      colnum = 0
      vector = []
      for col in row :
        if rownum == 0 :
          if colnum > 0 :
            X.append(float(col))
        else :
          if colnum == 0 :
            Y.append(float(col))
          else :
            if col == "NaN" :
              vector.append(np.nan)
            else :
              vector.append(float(col))
        colnum = colnum +1
      if rownum != 0 :
        Z.append(vector)
      rownum = rownum + 1
    X = np.array(X)
    Y = np.array(Y)
    X1, Y1 = np.meshgrid(X,Y)
    Z = np.ma.array(Z, mask=np.isnan(Z))
    return (X1, Y1, Z)

print 'Reading:', str(sys.argv[1])
gridSize = sys.argv[2]
windSize = sys.argv[3]

HisX,HisY,HisZ = read_2D(sys.argv[1])

noBin   = shape(HisX)[1]
maxZ    = np.nanmax(HisZ)
strd    = max(1,noBin/200)
zlim    = maxZ

fig = figure(figsize=(10,8))
ax = fig.gca(projection='3d')
ax.plot_surface(HisX, HisY, HisZ, vmax=zlim, rstride=strd, cstride=strd, linewidth=0, antialiased=False, alpha=0.2, cmap=cm.coolwarm)
ax.set_xticks(np.linspace(0,1,6))
ax.set_xlabel('x')
ax.set_yticks(np.linspace(0,1,6))
ax.set_ylabel('y')
ax.set_zlim3d(0,zlim)

interval        = zlim/20
levels          = np.arange(0, zlim, interval)
cset = ax.contour(HisX, HisY, HisZ, levels, zdir='z', offset=zlim*1.2, cmap=cm.coolwarm)

plt.tight_layout()

if True :
  fi=0
  nm="histograms/%s-%s.png" % (gridSize, windSize)
  savefig(nm)

# plt.show()

