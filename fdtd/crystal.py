#############
## MODULES ##
#############
import numpy as np
from time import sleep

##################
## DEPENDENCIES ##
##################
import tools as tls
from constants import c, mu0, eps0, eta0, k, e, hbarc

import scipy.linalg as lin

####################
## CRYSTALL CLASS ##
####################
class Crystal(object):

    ####################
    ## INITIALIZATION ##
    ####################
    def __init__(self, grid, shape, ul, fast_update=False):
        grid.objects.append(self)
        self.grid = grid
        self.du = grid.du
        m, n = self.ul = ul
        self.M, self.N = self.shape = shape
        
        self.locX = slice(m, m+self.M) # x-location of the crystal in the grid
        self.locY = slice(n, n+self.N) # y-location of the crystal in the grid
        self.m_min = m
        self.m_max = m+self.M
        self.n_min = n
        self.n_max = n+self.N

        self.grid.inv_eps[self.locX, self.locY, :] = 0.
        self.grid.inv_mu [self.locX, self.locY, :] = 0.
        
        self.update_E = self.update_E_fast if fast_update else self.update_E
        
        self.Ab = np.zeros((self.M, self.N)) # Absorbtion profile
        self.G = np.zeros((self.M, self.N)) # Generated free carriers
        
        self._r = np.zeros((6, 3))
        self._S = np.zeros((self.M, self.N, 3))
        
        self._material_parameters()

        # Dont change these...
        self.ND0_start = self.ND0
        self.ND = np.ones((self.M, self.N))*self.ND
        self.ND0 = np.ones((self.M, self.N))*self.ND0 # Filled trap density
        self.n = np.ones((self.M, self.N))*self.ND0*self.beta*self.tau # Dark free carrier density
        self.NDc = self.ND - self.ND0 - self.n

        self.A = tls.sparse_matrix(self.M, self.N)       
        self.x = np.zeros((2*self.M*self.N-self.M-self.N))
        self.update_S(solver=tls.solve)
                
    ################
    ## PARAMETERS ##
    ################
    def _material_parameters(self):
        pass # CREATE SUBCLASS WITH CORRECT PARAMETERS

    ################
    ## PROPERTIES ##
    ################
    
    ## WITH SPECIAL SETTERS
    @property
    def S(self):
        return self._S
    @S.setter
    def S(self, arr):
        self._S = arr
        self.update_inv_eps()
        
    @property
    def r(self):
        return self._r
    @r.setter
    def r(self, value):
        self._r = tls.pockels_tensor(value) if type(value)==dict else value
        self.update_inv_eps()
    
    @property
    def sigmaM(self):
        return self.alpha*eta0*self.grid.dt*np.sqrt(self.inv_eps)
        #ret[0,:,0] = 0.
        #ret[:,0,1] = 0.
        #return ret
    @sigmaM.setter
    def sigmaM(self,value):
        raise AttributeError("Setting sigmaM is not allowed. Consider setting alpha instead.")
        
    @property
    def eps(self):
        return (self._epsX, self._epsY, self._epsZ)
    @eps.setter
    def eps(self, value):
        if type(value) != tuple:
            value = (value, value, value)
        self._epsX = value[0]
        self._epsY = value[1]
        self._epsZ = value[2]
        self.update_inv_eps()

    ## DEPENDING ON BASE PARAMETERS
    @property
    def NDp(self): # Positive trap density
        return self.ND-self.ND0
    @NDp.setter
    def NDp(self,value):
        self.ND0 = self.ND-value
    @property
    def rho(self): # rho distribution
        return e*(self.NDp - self.n - self.NDc)
    @rho.setter
    def rho(self,value):
        raise AttributeError("Setting rho is not allowed.")

    @property
    def tau(self): # Free carrier lifetime
        return 1./(self.gamma*self.NDp) # [#/s]
    @tau.setter
    def tau(self,value):
        raise AttributeError("Setting tau is not allowed.")
    
    @property
    def sc(self):
        return 0.25*e*self.du/(k*self.T)
    @sc.setter
    def sc(self, value):
        raise AttributeError("Setting sc is not allowed")

    @property
    def dt(self): # Diffusion matched Lax-Friedrichs time step
        return 0.25*e*self.du**2/(self.mu*k*self.T)
    @dt.setter
    def dt(self,value):
        raise AttributeError("Setting dt is not allowed")

    @property
    def F(self):
        S = tls.H_Hz(self.S)
        S[ 0, :] = 0
        S[-1, :] = 0
        S[:,  0] = 0
        S[:, -1] = 0        
        return S*self.n[..., None]
    @F.setter
    def F(self,value):
        raise AttributeError("Setting F is not allowed")
    
    @property
    def E(self):
        return self.grid.E[self.locX,self.locY]
    @E.setter
    def E(self, value):
        self.grid.E[self.locX,self.locY] = value

    @property
    def index(self):
        return np.sqrt(1./self.inv_eps[...,2])

    @property
    def index_variation(self):
        return self.index - self.index.mean()
    
    ##################
    ## FDTD UPDATES ##
    ##################
    def update_E_fast(self, curl_H): # Update equation for the electric field WITHOUT TE and TM component mixing --> Much faster!
        self.grid.E[self.locX, self.locY, :] += self.grid.sc * self.inv_eps * curl_H

    def update_E(self, curl_H):
        M = self.M
        N = self.N
        
        curl_H_z = tls.E_Ez(curl_H)
        self.grid.E[self.locX, self.locY, 2] += self.grid.sc*(tls.matrix_multiply(self.inv_epsZ, curl_H_z))[..., 2]
        
        curl_H_z_y = curl_H_z.copy()
        curl_H_z_y[...,1] = 0
        curl_H_y = np.zeros((M, N, 3))
        curl_H_y[...,1] += curl_H[..., 1]       
        self.grid.E[self.locX, self.locY, 1] += self.grid.sc*(tls.Ez_Ey(tls.matrix_multiply(self.inv_epsZ, curl_H_z_y)) + tls.matrix_multiply(self.inv_epsY, curl_H_y))[..., 1]
      
        curl_H_z_x = curl_H_z.copy()
        curl_H_z_x[...,0] = 0
        curl_H_x = np.zeros((M, N, 3))
        curl_H_x[...,0] += curl_H[..., 0]
        self.grid.E[self.locX, self.locY, 0] += self.grid.sc*(tls.Ez_Ex(tls.matrix_multiply(self.inv_epsZ, curl_H_z_x)) + tls.matrix_multiply(self.inv_epsX, curl_H_x))[..., 0]

                
    def update_H(self, curl_E): # Assuming the magnetic permeability is 1 throughout the crystal
        f = 0.5 * self.sigmaM
        Hpre = self.grid.H[self.locX,self.locY].copy()
        self.grid.H[self.locX,self.locY] = Habs = (1-f)/(1+f)*Hpre - self.grid.sc/(1+f)*curl_E
        Hnoabs = Hpre - self.grid.sc*curl_E

        # ABSORPTION CALCULATION
        # we take average of x and y component of H field as the alternative for the Ez component.
        Hnoabs[...,1] += Hnoabs[...,0]
        Hnoabs[...,1]/=2.
        Hnoabs[...,0] = 0
        Habs[...,1] += Habs[...,0]
        Habs[...,1]/=2.
        Habs[...,0] = 0
        dG = 0.5*tls.sumc(Hnoabs**2-Habs**2)
        self.G += self.qe*dG*self.grid.wl/hbarc*self.ND0/self.ND0_start
    

    ########################
    ## MAIN ELECTRON LOOP ##
    ########################
    def run_diffusion(self, Q, extrapolation = None):
        n0 = self.n.copy()
        for q in xrange(Q):
            self.diffuse()
        return self.n - n0
            
    
    def diffuse(self):  
        dn = self.G + self.beta*self.Dt*self.ND0 # Be careful that the [adjusted] fdtd time is equal to Dt!
        self.n += dn
        self.ND0 -= dn

        assert self.Dt >= self.dt
        N = min(25, int(self.Dt/self.dt+0.5)) # We are allowed to do make this cutoff, as equilibrium is reached fast
        for i in xrange(N):
            self.update_n()
        
        dn = -self.n*self.Dt/self.tau
        self.n += dn
        assert (self.n >= 0.).all()
        self.ND0 -= dn

        self.update_S()
        
    ######################
    ## ELECTRON UPDATES ##
    ######################
    def update_n(self):
        Fx,Fy,_ = self.F.transpose(2,0,1)
        self.n = 0.5*(tls.savx(self.n) + tls.savy(self.n)) + self.sc*(tls.sdx(Fx) + tls.sdy(Fy))
            
    def update_S(self, solver=tls.bicgstab):
        M = self.M
        N = self.N
        rl = self.du*self.rho.copy()/(self.eps_static*eps0)
        b = np.zeros((2*M*N-M-N))
        b[:M*N-1] = rl.ravel()[1:]
        
        self.x = solver(self.A, b, x0=self.x, tol=0.01, M=self.A.T)[0]
        
        x1, x2 = np.split(self.x, [M*N-N])
        
        S = np.zeros((M,N,3))
        S[1:, :, 0] = x1.reshape((M-1, N))
        S[:, 1:, 1] = x2.reshape((M, N-1))
        
        self.S = S
        
    ###################
    ## OTHER UPDATES ##
    ###################
    def update_inv_eps(self): 
        self.inv_epsX = tls.inv_epsilon(tls.H_Ex(self.S), self.eps, self.r)
        self.inv_epsY = tls.inv_epsilon(tls.H_Ey(self.S), self.eps, self.r)
        self.inv_epsZ = tls.inv_epsilon(tls.H_Ez(self.S), self.eps, self.r)
        self.inv_eps = np.zeros((self.M, self.N, 3))
        self.inv_eps[..., 0] = self.inv_epsX[...,0,0]
        self.inv_eps[..., 1] = self.inv_epsX[...,1,1]
        self.inv_eps[..., 2] = self.inv_epsX[...,2,2]

    ###########
    ## OTHER ##
    ###########
    def reset(self):
        self.G[:,:] = 0

    def plot(self, ax=None):
        from matplotlib.patches import Rectangle
        if ax is None:
            from matplotlib.pyplot import gca
            ax = gca()
        ax.add_patch(Rectangle((self.ul[1]-0.5,self.ul[0]-0.5), self.N, self.M, hatch='\\', alpha=0.4, color='green'))
        
    def save(self):
        tls.cd(self.name)
        np.save(name+'shape',self.shape)
        np.save(name+'ul',self.ul)
        np.save(name+'beta',self.beta)
        np.save(name+'gamma',self.gamma)
        np.save(name+'T',self.T)
        np.save(name+'Dt',self.Dt)
        np.save(name+'eps_static',self.eps_static)
        np.save(name+'S',self.S)
        np.save(name+'r',self.r)
        np.save(name+'eps',self.eps)
        np.save(name+'alpha',self.alpha)
        np.save(name+'ND',self.ND)
        np.save(name+'ND0',self.ND0)
        np.save(name+'n',self.n)


###############
## MATERIALS ##
###############
class LiNbO3(Crystal):      
    def _material_parameters(self):
        self.beta = 1. #[1/s] Thermal carrier excitation rate

        self.qe = 1. # quantum efficiency of exciting electrons by absorbed photons
                
        self.gamma = 1e-17 #[m3/s] Regeneration rate. Will be combined with NDp into tau, the carrier lifetime
        
        self.mu = 0.0005 #[m2/Vs] Carrier mobility. Is only responsible for the time step of the diffusion!
        self.T = 300 #[K] Temperature: inversely proportional with the courant number and time step
        
        self.Dt = 1e-8 #[s] Secondary time step for generation and recombination of free carriers.

        self.eps_static = 32 # Relative permittivity for static fields (like the space-charge fg inield)
                   
        # Sellmeier equation
        wl = self.grid.wl*np.sqrt(self.grid.eps)*1e6 #[um]   
        A,B,C,D = (4.9048,0.11768,-0.0475,-0.027169)
        epso = A + B/(wl*wl+C) + D*wl*wl
        A,B,C,D = (4.582,0.099169,-0.044432,-0.02195)
        epse = A + B/(wl*wl+C) + D*wl*wl        
        self.eps = (epso, epso, epse) # Diagonal relative permittivity element(s) for the lightfield

        # Pockels tensor for a 3mm material
        # with optical axis in the y direction
        a,b,c,d = 10e-12, 6.8e-12, 32.2e-12, 32e-12
        r = np.array([ [b , a , 0.],    # Variation in index for Ex due to Sx and Sy
                       [0., c , 0.],    # Variation in index for Ey due to Sy
                       [-b, a , 0.],    # Variation in index for Ez due to Sx and Sy
                       [0., 0., d ],    # Slight rotation between Ez and Ey due to Sz -> No rotation, bc no Sz in 2D
                       [0., 0.,-b ],    # Slight rotation between Ez and Ex due to Sz -> No rotation, bc no Sz in 2D
                       [d , 0., 0.] ])  # Strong rotation between Ex and Ey due to Sx
        self.r = r
    
        self.S = np.zeros((self.M, self.N, 3)) # Space rho Electric Field
    
        self.alpha = 20. # (Arbitrary) Absorption in the crystal

        ## TRAP AND CARRIER DENSITIES
        self.ND = 6.6e24 #[1/m3] # Total number of traps
        self.ND0 = 3.3e24 #[1/m3] # Number of neutral traps
