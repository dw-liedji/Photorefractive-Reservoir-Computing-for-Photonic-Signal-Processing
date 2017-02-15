
# coding: utf-8

# # Modules

# In[1]:

import sys
import os
LIBS_PATH = os.environ.get('PYTHON_LIBS')
if LIBS_PATH is not None:
    sys.path.append( LIBS_PATH )

from timeit import default_timer as time
from fdtd import *
import numpy as np
from fdtd import plotter as plt


# # Initialization

# Assign the number of threads.
Np = 1 if len( sys.argv ) <= 1 else int( sys.argv[1] )
# Assign the size of the matrix.
SIZE = 300 if len( sys.argv ) <= 2 else int( sys.argv[2] )
# Assign the number of iterations.
Q = 20 if len( sys.argv ) <= 3 else int( sys.argv[3] )

CU = SIZE
M  = SIZE
R  = SIZE
CO = 3

print "Starting Crystal.py with parameters:"
print "Matrix: ",CU,"x",M,"x",R,"x",CO
print "Q = ",Q
print "Threads = ",Np

print "You can invoke it with: python Crystal.py [THREADS] [SIZE] [ITERATIONS]"

# ## Grid

# We create a grid in the same way as before

# In[2]:

tt=time()
grd = grid.Grid((CU,M,R,CO), pml_thickness=10)

# ## Source

# We define the length of the pulse as

# In[3]:

# We create two sources this time:

# In[4]:

lft = source.Oblique(grd, (CU/2,100), size=CU/2, tan=0, period=20, phi=0, I=1e4, 
                     pulselength=Q, sigma=10, r=4, mode='TE')
top = source.Oblique(grd, (CU/2,100), size=CU/2, tan=1e20, period=20, phi=0, I=1e4, 
                     pulselength=Q, sigma=10, r=4, mode='TE')


# ## Detectors

# In[5]:

rgt = detector.VerticalDetector(grd, grd.N-10)
btm = detector.HorizontalDetector(grd, grd.M-10)


# ## Crystal

# It is easy to include a LiNbO3 crystal in the grid. We specify the location of the upper left corner (ul) and the shape (dimensions) fo the crystal. The rest is already specified in the material parameters of LiNbO3 in the crystals module.

# In[6]:

crst = crystal.LiNbO3(grd, ul=(11,11), shape=(200,200))
crst.r = crst.r

print "Setup time: " , time()-tt


# ## Show setup:

# In[7]:

grd.plot()


# Where as before sources are indicated by red lines and detectors by light blue lines. Crystals are indicated by a light green shaded area.

# # Run

# We run the propagation of the light for Q timesteps.

# In[8]:
t1=time()
grd.run_fdtd( Q, Np )
print time()-t1

# # Visualize [1]

# We choose to visualize the generated electrons in the crystal.

# In[10]:

plt.imshow(crst.G)
plt.show()


# # Diffusion

# Exciting carriers in the crystal gives rise to a refractive index distribution. However, at the vastly different timescales that this happens, combining both is not always straightforward. Below is an example on how one might approach this problem:

# ## FDTD

# We already ran the FDTD simulation for about 3e-13s.

# ## Timescale compensation

# We recognize three timescales in this problem:
# 1. The timescale of the propagation of the light through the crystal: `Q*grd.dt`
# 2. The timestep of the diffusion: `crst.dt`
# 3. The timestep of the electron generation and recombination: `crst.Dt`
# 
# The first and the second one are already fixed by the courant number and the lax-friedrich scheme update scheme respectively.

# In[11]:

print "1. Light propagation time", Q*grd.dt
print "2. Carrier diffusion time step", crst.dt, "s"


# The third one is free to choose. For this example, we choose

# In[12]:

crst.Dt = 1e-8 #[s]
print "3. Carrier generation time step", crst.Dt, "s"


# Before we sgrd = grid.Grid((121,121), pml_thickness=10,eps=4.84)tart the diffusion, we choose to multiply the the generated electrons by a multiplication factor corresponding to mismatch between the propagation time an the electron generation timestep `crst.Dt`:

# In[13]:

f = crst.Dt/(Q*grd.dt)
crst.G *= f


# This is quite allowed, as we can assume that the refractive index of the crystal remains constant during one diffusion timestep.

# ## Diffusion

# We will run the diffusion over R timesteps:

# In[14]:

R = 1


# After resolving the timescale mismatches, we can finally proceed to do a diffusion loop:

# In[15]:

crst.run_diffusion(R)
print "Cristal diffusion elapsed time: ", R*crst.Dt, "s"


# ## Note

# Note that the speed of the diffusion depends a lot on a correct choice of the diffusion parameters. A too small generation time step will make the simulations a lot slower. However, this timestep can also not be taken too big to keep the simulations numerically stable.

# # visualize [2]

# The resulting refractive index variation can be shown:

# In[16]:

print "Totale Completion Time :" , time()-t, "Total propagation time: ", t2
#plt.imckness=10)
show(crst.index_variation)
plt.show()
