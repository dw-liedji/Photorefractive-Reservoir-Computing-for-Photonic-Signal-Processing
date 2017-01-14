
# coding: utf-8

# # Modules

# In[1]:

from timeit import default_timer as time
from fdtd import *
#get_ipython().magic(u'matplotlib inline')
import numpy as np
from fdtd import plotter as plt


# # Initialization

# ## Grid

# We create a grid by specifying its dimensions. Optional arguments include the pml thickness (for the absorbing boundaries) and the permittivity in the grid (square of the refractive index). Periodic boundaries can also be included by specifying the periodic boundaries flag.

CU = 300
M  = 300
R  = 300
CO = 3

Q  = 20

print "Matrix: ",CU,"x",M,"x",R,"x",CO
print "Q = ",Q

# In[2]:
tt=time()
grd = grid.Grid((CU,M,R,CO), pml_thickness=10)


# ## Source

# A source can also be included. All parameters are always given in timesteps [dt] and gridpoints [du].

# In[3]:

src = source.Oblique(grd, (CU/2,50), size=CU, tan=0, period=20, phi=0, I=1e4, 
                     pulselength=360, sigma=10, r=4, mode='TE')


src2 = source.Oblique(grd, (CU/2,100), size=CU/2, tan=0, period=20, phi=0, I=1e4,
                     pulselength=160, sigma=10, r=4, mode='TE')

print "Setup time: " , time()-tt

# # Run [1]

# The FDTD can be run in two different ways:

# In[4]:
Np = 24
for n in xrange(1,Np+1):
  t1=time()
  grd.run_fdtd(Q,n)
  print time()-t1

# In[5]:

#for i in xrange(100):
#    grd.step_fdtd()


# The second one is sometimes preferred if a progress bar is wanted. (for example by using the tqdm package):
# ```python
# from tqdm import tqdm
# for i in tqdm(xrange(100)):
#     grd.step_fdtd()
# ```

# # visualization [1]

# In[6]:

#from fdtd import plotter as plt


# Visualization can be done by the plotter package, which is just a `matplotlib.pyplot` wrapper:

# In[7]:

plt.imshow(grd.E[...,2])
plt.show()


# # Reset

# Sometimes it is necessary to reset to initial conditions. This can be done by the reset, function of the grid class, which sets the fields to zero, as well as the internal time parameter `grd.q`. If there are any objects in the grid, these get reset as well.

# In[8]:

grd.reset()


# # Detectors

# Adding a detector to the grid is simple. There are two kind of detectors: Horizontal detectors and Vertical detectors:

# In[9]:

rgt = detector.VerticalDetector(grd, grd.N-10)
btm = detector.HorizontalDetector(grd, grd.M-10)


# # Run [2]

# In[10]:
#t2=time()
#grd.run_fdtd(1)
#print "Test 2 : ", time()-t2

# # Visualization [2]

# Another way to visualize the fields is by using the `grid.plot` function. Which shows all the objects in the grid, as well as the $E_z$ component of the electromagnetic field.

# In[11]:

grd.plot()


# Where Sources are indicated by red lines and detectors by light blue lines.

# # Detected Fields

# We can show the detected field intensity, by taking the sum of the square of the detected fields:

# In[1]:

plt.plot(sum(rgt.E[:,:,2])**2)
plt.show()


# Where the sum was taken over all timesteps `q` of the field intensity of the `Ez` component.




