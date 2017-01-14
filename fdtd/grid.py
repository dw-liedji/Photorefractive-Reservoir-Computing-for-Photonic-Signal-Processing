#############
## MODULES ##
#############

import numpy as np
from scipy import sparse
from multiprocessing import Pool

##################
## DEPENDENCIES ##
##################
import pml
import tools as tls
from constants import c #[m/s] speed of light

################
## GRID CLASS ##
################
class Grid():
    def __init__(self, shape, wl=632.8e-9, Nwl=10, eps=1., pml_thickness = 10, periodic_boundaries = False):
        self.sc = 0.7 # Courant Number
        self.du = wl/Nwl # Grid spacing
        self.wl = wl
        self.Nwl = Nwl
        self.dt = self.sc*self.du/c # Time step
        M, N = shape
        self.M = M
        self.N = N
        
        self.E = np.zeros((M,N,3))
        self.H = np.zeros((M,N,3))
        
        self.inv_eps = np.ones((M,N,3))/eps # Diagonal matrix
        self.eps = eps
        
        self.inv_mu  = np.ones((M,N,3)) # Diagonal matrix
        # Note that locations of inv_mu and inv_eps where crystals are added are set to zero
        
        self.sources  = []
        self.objects  = []
        self.detectors= []

        self.q = 0

        self.pml = pml.PML(self, pml_thickness) if pml_thickness else pml.noPML(self)
        
        self.periodic_boundaries = periodic_boundaries
        
    def run_fdtd(self, steps):
        for q in xrange(steps):
            self.step_fdtd()

    def step_fdtd(self):
        self.update_E()
        for source in self.sources:
            source.E(self.q, self.E)
        self.update_H()
        for source in self.sources:
            source.H(self.q, self.H)
        self.q += 1

    def update_E(self):
        self.pml.update_phi_E()    
        curl_H = tls.curl_H(self.H)
        
        self.E += self.sc * self.inv_eps * curl_H

        for obj in self.objects:
            obj.update_E(curl_H[obj.locX, obj.locY])
            
        for det in self.detectors:
            det.detect_E()

        if self.periodic_boundaries:
            self.E[0,:] = self.E[-1,:]
        
        self.pml.update_E()
                
    def update_H(self):
        self.pml.update_phi_H()

        curl_E = tls.curl_E(self.E)
        
        self.H -= self.sc * self.inv_mu * curl_E

        for obj in self.objects:
            obj.update_H(curl_E[obj.locX, obj.locY])
                    
        if self.periodic_boundaries:
            self.H[-1,:] = self.H[0,:]
        else: # Make grid symmetric
            self.H[-1,:, 1] = 0 
            self.H[:,-1, 0] = 0
            self.H[-1,-1,2] = 0  

        self.pml.update_H()

    def reset(self):
        self.H[:,:,:] = 0.
        self.E[:,:,:] = 0.
        self.q = 0
        for obj in self.objects:
            obj.reset()
        for det in self.detectors:
            det.reset()

    def plot(self):
        from matplotlib.pyplot import subplots, show, grid, close, savefig
        fig, ax = subplots(1,1)
        grid()
        ax.set_ylim(self.M,0)
        ax.set_xlim(0,self.N)

        # init
        _ = np.zeros((self.M, self.N))
        _[0,0] += 1.
        _[0,1] -= 1.
        img = ax.imshow(_, cmap='RdBu', interpolation='none')

        # Ez comp.
        p = self.E[...,2].copy()
        if (p == 0).all():
            p = np.zeros((self.M,self.N))
        else:
            p /= np.abs(p).max()

        # show pml
        p += 0.66*self.pml.loc
        img.set_data(p)
        
        # show the rest
        for src in self.sources:
            src.plot(ax)
        for det in self.detectors:
            det.plot(ax)
        for obj in self.objects:
            obj.plot(ax)
        savefig('grd')
        show()
