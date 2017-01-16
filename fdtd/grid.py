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
    def __init__(self, shape, Np, wl=632.8e-9, Nwl=10, eps=1., pml_thickness = 10, periodic_boundaries = False):
        self.sc = 0.7 # Courant Number
        self.du = wl/Nwl # Grid spacing
        self.wl = wl
        self.Nwl = Nwl
        self.dt = self.sc*self.du/c # Time step
        CU, M, N, CO = shape
        self.CU = CU
        self.M = M
        self.N = N
        self.CO = CO

        self.curl = np.zeros((CU,M,CO)).astype(dtype=np.float32)

        self.E_3D = np.zeros((CU,M,N,CO)).astype(dtype=np.float32)
        self.H_3D = np.zeros((CU,M,N,CO)).astype(dtype=np.float32)

        self.E = self.E_3D[:,:,1,:]
        self.H = self.H_3D[:,:,1,:]

        self.inv_eps_3D = np.ones((CU,M,N,CO)).astype(dtype=np.float32)/eps # Diagonal matrix
        self.inv_eps = self.inv_eps_3D[:,:,1,:]
        self.eps = eps

        self.inv_mu_3D  = np.ones((CU,M,N,CO)).astype(dtype=np.float32) # Diagonal matrix
        self.inv_mu  = self.inv_mu_3D[:,:,1,:]
        # Note that locations of inv_mu and inv_eps where crystals are added are set to zero
        
        self.sources  = []
        self.objects  = []
        self.detectors= []

        self.q = 0

        self.pml = pml.PML(self, pml_thickness) if pml_thickness else pml.noPML(self)
        
        self.periodic_boundaries = periodic_boundaries
        
        init( Np )
        
    def run_fdtd(self, steps, N ):
        for q in xrange(steps):
            self.step_fdtd(N)

    def step_fdtd(self, N):
        self.update_E(N)
        for source in self.sources:
            source.E(self.q, self.E)
        self.update_H(N)
        for source in self.sources:
            source.H(self.q, self.H)
        self.q += 1

    def update_E(self):
        self.pml.update_phi_E()    
        
        #curl_H = tls.curl_H(self.H)
        #self.E += self.sc * self.inv_eps * curl_H
        
        curl_E( self.E_3D, self.H_3D, self.inv_eps_3D, self.curl, self.sc, N )

        for obj in self.objects:
            obj.update_E(curl_H[obj.locX, obj.locY])
            
        for det in self.detectors:
            det.detect_E()

        if self.periodic_boundaries:
            self.E[0,:] = self.E[-1,:]
        
        self.pml.update_E()
                
    def update_H(self, N):
        self.pml.update_phi_H()

        #self.curl = tls.curl_E_3D(self.E_3D)
        #self.H -= self.sc * self.inv_mu * self.curl

        curl_E( self.E_3D, self.H_3D, self.inv_mu_3D, self.curl, self.sc, N )

        for obj in self.objects:
            obj.update_H(self.curl[obj.locX, obj.locY])

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
