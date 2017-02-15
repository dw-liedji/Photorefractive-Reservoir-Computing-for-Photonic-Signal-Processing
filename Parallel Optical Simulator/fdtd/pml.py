#############
## MODULES ##
#############
import numpy as np
import tools as tls
from multiprocessing import Pool

###################
## PML STRUCTURE ##
###################
class noPML_single(object):
    def __init__(self, grid):
        self.loc = np.zeros((grid.M,grid.N), dtype=bool)
    def update_E(self):
        pass
    def update_H(self):
        pass     
    def update_phi_E(self):
        pass
    def update_phi_H(self):
        pass

######################
## SINGLE PML CLASS ##
######################
class PML_single(noPML_single):
    
    def __init__(self, grid, thickness = 10, orientation = 0):
        self.grd = grid
        self.M = grid.M
        self.N = grid.N
        self.vectE = ((np.arange(thickness,dtype=float)+0.5)**3)/(thickness+1)**4*10.*4 
        self.vectH = ((np.arange(thickness+1,dtype=float))**3)/(thickness+1)**4*10.*4

    def update_E(self):
        self.grd.E[self.loc] += self.grd.sc * self.grd.inv_eps[self.loc] * self.phiE

    def update_H(self):
        self.grd.H[self.loc] -= self.grd.sc * self.grd.inv_mu[self.loc] * self.phiH
                   
    def update_phi_E(self):
        self.psi_Ex[...,1] = self.psi_Ex[...,1]*self.bE[...,1] + tls.dHz_Ex(self.grd.H[self.locz])*self.cE[...,1]
        self.psi_Ey[...,0] = self.psi_Ey[...,0]*self.bE[...,0] + tls.dHz_Ey(self.grd.H[self.locz])*self.cE[...,0]
        self.psi_Ez[...,0] = self.psi_Ez[...,0]*self.bE[...,0] + tls.dHy_Ez(self.grd.H[self.locy])*self.cE[...,0]
        self.psi_Ez[...,1] = self.psi_Ez[...,1]*self.bE[...,1] + tls.dHx_Ez(self.grd.H[self.locx])*self.cE[...,1]
        self.phiE = np.array([self.psi_Ex[...,1],-self.psi_Ey[...,0],self.psi_Ez[...,0]-self.psi_Ez[...,1]]).transpose(1,2,0)
    def update_phi_H(self):
        self.psi_Hx[...,1] = self.psi_Hx[...,1]*self.bH[...,1] + tls.dEz_Hx(self.grd.E[self.locz])*self.cH[...,1]
        self.psi_Hy[...,0] = self.psi_Hy[...,0]*self.bH[...,0] + tls.dEz_Hy(self.grd.E[self.locz])*self.cH[...,0]
        self.psi_Hz[...,0] = self.psi_Hz[...,0]*self.bH[...,0] + tls.dEy_Hz(self.grd.E[self.locy])*self.cH[...,0]
        self.psi_Hz[...,1] = self.psi_Hz[...,1]*self.bH[...,1] + tls.dEx_Hz(self.grd.E[self.locx])*self.cH[...,1]
        self.phiH = np.array([self.psi_Hx[...,1],-self.psi_Hy[...,0],self.psi_Hz[...,0]-self.psi_Hz[...,1]]).transpose(1,2,0)
        
    def set_constants(self):
        self.aE += 1e-8 # for numerical stability (removes divide by 0 errors)
        self.aH += 1e-8 # for numerical stability (removes divide by 0 errors)
        self.kE = 1.2
        self.kH = 1.2
    
        self.bE = np.exp(-(self.sigmaE/self.kE+self.aE)*self.grd.sc)
        self.bH = np.exp(-(self.sigmaH/self.kH+self.aH)*self.grd.sc)
    
        self.cE = (self.bE-1.)*self.sigmaE / (self.sigmaE*self.kE+self.aE*self.kE**2)
        self.cH = (self.bH-1.)*self.sigmaH / (self.sigmaH*self.kH+self.aH*self.kH**2)
    
        #del self.sigmaE
        #del self.sigmaH
        del self.M
        del self.N

    
    def initialize(self):
        self.sigmaE = np.zeros((self.M, self.N, 2))
        self.sigmaH = np.zeros((self.M, self.N, 2))
        self.aE = np.zeros((self.M, self.N, 2)) 
        self.aH = np.zeros((self.M, self.N, 2)) 
        self.psi_Ex = np.zeros((self.M, self.N, 2))
        self.psi_Ey = np.zeros((self.M, self.N, 2))
        self.psi_Ez = np.zeros((self.M, self.N, 2))
        self.psi_Hx = np.zeros((self.M, self.N, 2))
        self.psi_Hy = np.zeros((self.M, self.N, 2))
        self.psi_Hz = np.zeros((self.M, self.N, 2))
            
 
###########################
## SINGLE PML SUBCLASSES ##
###########################
class PML_xlow(PML_single):
    def __init__(self, grid, thickness = 10):
        PML_single.__init__(self, grid, thickness)
        self.M = thickness
        
        self.loc = (slice(None,thickness),slice(None,None))
        self.locx = (slice(None,thickness),slice(None,None),0)
        self.locy = (slice(None,thickness),slice(None,None),1)
        self.locz = (slice(None,thickness),slice(None,None),2)        
        
        self.initialize()
        self.sigmaE[..., 0] = np.array([(self.vectE[::-1])]*self.N).T
        self.sigmaH[:thickness, :, 0] = np.array([self.vectH[-2::-1]]*self.N).T
        self.set_constants()        

class PML_xhigh(PML_single):
    def __init__(self, grid, thickness = 10):
        PML_single.__init__(self, grid, thickness)  
        self.M = thickness
        
        self.loc = (slice(-thickness,None),slice(None,None))
        self.locx = (slice(-thickness,None),slice(None,None),0)
        self.locy = (slice(-thickness,None),slice(None,None),1)
        self.locz = (slice(-thickness,None),slice(None,None),2)        
        
        self.initialize()
        self.sigmaE[..., 0] = np.array([self.vectE]*self.N).T
        self.sigmaH[-thickness:-1,:, 0] = np.array([self.vectH[1:-1]]*self.N).T
        self.set_constants()  
        
class PML_ylow(PML_single):
    def __init__(self, grid, thickness = 10):
        PML_single.__init__(self, grid, thickness)   
        self.N = thickness
        
        self.loc = (slice(None,None),slice(None,thickness))
        self.locx = (slice(None,None),slice(None,thickness),0)
        self.locy = (slice(None,None),slice(None,thickness),1)
        self.locz = (slice(None,None),slice(None,thickness),2)        
        
        self.initialize()
        self.sigmaE[..., 1] = np.array([self.vectE[::-1]]*self.M)
        self.sigmaH[:,:thickness, 1] = np.array([self.vectH[-2::-1]]*self.M)
        self.set_constants()

class PML_yhigh(PML_single):
    def __init__(self, grid, thickness = 10):
        PML_single.__init__(self, grid, thickness)            
        self.N = thickness

        self.loc = (slice(None,None),slice(-thickness,None))
        self.locx = (slice(None,None),slice(-thickness,None),0)
        self.locy = (slice(None,None),slice(-thickness,None),1)
        self.locz = (slice(None,None),slice(-thickness,None),2)        
        
        self.initialize()
        self.sigmaE[..., 1] = np.array([self.vectE]*self.M)
        self.sigmaH[:,-thickness:-1, 1] = np.array([self.vectH[1:-1]]*self.M)
        self.set_constants()        
        
#####################
## PML DUMMY CLASS ##
##################### 
#This PML is used if we choose to have no absorbing boundaries.
class noPML(object):
    def __init__(self,grid):
        self.m_max = grid.M
        self.m_min = 0
        self.n_max = grid.N
        self.n_min = 0
    def update_phi_E(self):
        pass
    def update_phi_H(self):
        pass     
    def update_E(self):
        pass        
    def update_H(self):
        pass

####################
## MAIN PML CLASS ##
#################### 

    
#A PML is a collection of 4 single PMLs (6 in 3D) that surround the region of interest.
class PML(noPML):
    def __init__(self, grid, thickness=10):
        if type(thickness) == int:
            thickness = (thickness,thickness,thickness,thickness)
        self.xlow = PML_xlow(grid, thickness[0]) if thickness[0] else noPML_single(grid)
        self.xhigh = PML_xhigh(grid, thickness[1]) if thickness[1] else noPML_single(grid)
        
        self.ylow = PML_ylow(grid, thickness[2]) if thickness[2] else noPML_single(grid)
        self.yhigh = PML_yhigh(grid, thickness[3]) if thickness[3] else noPML_single(grid)
        
        self.m_min = thickness[0]        
        self.m_max = grid.M-thickness[1]
        self.n_min = thickness[2]        
        self.n_max = grid.N-thickness[3]
        
        self.loc = np.zeros((grid.M,grid.N), dtype=bool)
        self.loc[self.xlow.loc ] = True
        self.loc[self.xhigh.loc] = True
        self.loc[self.ylow.loc ] = True
        self.loc[self.yhigh.loc] = True
    
    def update_phi_E(self):
        self.xlow.update_phi_E()
        self.yhigh.update_phi_E()
        self.xhigh.update_phi_E()
        self.ylow.update_phi_E()
        
    def update_phi_H(self):
        self.xlow.update_phi_H()
        self.yhigh.update_phi_H()
        self.xhigh.update_phi_H()
        self.ylow.update_phi_H() 
        
    def update_E(self):
        self.xlow.update_E()
        self.yhigh.update_E()
        self.xhigh.update_E()
        self.ylow.update_E()  
        
    def update_H(self):
        self.xlow.update_H()
        self.yhigh.update_H()
        self.xhigh.update_H()
        self.ylow.update_H()
        
