#############
## MODULES ##
#############
import numpy as np
from matplotlib.pyplot import plot

##################
## DEPENDENCIES ##
##################
from constants import c, eps0

######################
## USEFUL FUNCTIONS ##
######################

## Envelope
def pulse(pulselength, sigma, r=4): # __/''''''''\__
    pulse = np.ones(pulselength)
    q = np.arange(pulselength)
    loc = q<r*sigma 
    pulse[loc] = np.exp(-0.5*(q[loc]-r*sigma)**2/sigma**2)
    loc = q > pulselength-r*sigma
    pulse[loc]=np.exp(-0.5*(q[loc]-pulselength + r*sigma)**2/sigma**2)
    return pulse

#################
## BIT CLASSES ##
#################

class Zero(object):
    def __getitem__(self, q):
        return 0.

class One(object):
    def __getitem__(self, q):
        return 1.

class Stream(object):
    def __init__(self, bits, pulse, bitlength=None):
        self.bits = bits
        self.bitlength = max(len(pulse), 0 if bitlength is None else bitlength)
        self.pulse = np.hstack((pulse, np.zeros(self.bitlength-len(pulse))))
    def __getitem__(self, q):
        return self.pulse[q%self.bitlength] if self.bits[int(q/self.bitlength)] else 0.

class Pulse(object):
    def __init__(self, pulse):
        self.pulse = pulse
        self.bitlength = len(pulse)
    def __getitem__(self, q):
        return self.pulse[q] if q < self.bitlength else 0.

##################
## PARENT CLASS ##
##################

class Source(object):  
    def H(self, q, H):
        pass
    def E(self, q, E):
        pass
    def plot(self, ax=None):
        if ax is None:
            from matplotlib.pyplot import gca
            ax = gca()
        y,x = self.loc()
        ax.plot(np.array(x),np.array(y), color='red')

#################
## MAIN SOURCE ##
#################

class Oblique(Source):    
    def __init__(self, grid, center, size, tan, period, phi=0, I=1e4, pulselength=800, sigma = 40, r = 4, mode="TE", hard=False, bits = [], bitlength=None):
        grid.sources.append(self)
        self.grid = grid
        self.period = period
        self.size = size # Length of the source generating line [in gridpoints]
        self.phi = phi # phasedelay
        self.amp = np.sqrt(72*I*np.sqrt(grid.inv_eps[center[0],center[1],2])/(np.pi*c)) # Amplitude of the source corresponding to the requested intensity

        self.sigma = sigma # Number of timesteps in a sigma
        self.r = r # Number of sigmas in rising edge of the gaussian pulse (in time); usually 3 or 4
        self.pulselength = pulselength # Number of timesteps of the pulse (if pulselength>2r*sigma then the rest of the pulse is set to 1.)
        self.pulse = pulse(pulselength, sigma, r)
               
        # Location of the upper left corner
        self.m = center[0] - int(0.5*size)
        self.n = center[1] - int(0.5*size)

        # Different update equation for hard sources
        if hard:
            self.Fx = self.Fx_hard
            self.Fy = self.Fy_hard
        
        # Different update equation for tan > 1 or TE/TM
        if abs(tan) <= 1:
            self.tan = tan
            self.loc = self.loc_x
            if mode == "TE":
                self.E = self.Fx
            elif mode == "TM":
                self.H = self.Fx
            elif mode == "TEM" or mode == "TETM":
                self.E = self.H = self.Fx
        else:
            self.tan = 1./tan
            self.loc = self.loc_y
            if mode == "TE":
                self.E = self.Fy
            elif mode == "TM":
                self.H = self.Fy
            elif mode == "TEM" or mode == "TETM":
                self.E = self.H = self.Fy
        # Stretchfactor to correct for the fact that taking a step under an angle
        # is longer by a factor 1 / cos than a step along the grid
        stretchfactor = np.sqrt(1+self.tan**2) # = 1 / cos
        
        self.vect = (np.arange(self.size)-float(self.size)/2+0.5)*stretchfactor
        self.profile = np.exp(-self.vect**2/(2*(self.size/5)**2))

        if bits != []: # For an information carrying signal:
            self.stream = Stream(bits, self.pulse, bitlength)
        else: # Just one pulse
            self.stream = Pulse(self.pulse)
        
    def Fx(self, q, F):
        vect = self.amp*np.sin(2*np.pi*q/self.period + self.phi)*self.stream[q]*self.profile
        for m in xrange(self.size):
            n = round(self.tan*(m-self.size/2) + self.size/2)
            if abs(self.m+m) < self.grid.M and abs(self.n+n) < self.grid.N:
                F[self.m+m, self.n+n, 2] += vect[m]
                
    def Fy(self, q, F):
        vect = self.amp*np.sin(2*np.pi*q/self.period + self.phi)*self.stream[q]*self.profile
        for n in xrange(self.size):
            m = round(self.tan*(n-self.size/2) + self.size/2)
            if abs(self.m+m) < self.grid.M and abs(self.n+n) < self.grid.N:
                F[self.m+m, self.n+n, 2] += vect[n]

    def Fx_hard(self, q, F):
        vect = self.amp*np.sin(2*np.pi*q/self.period + self.phi)*self.stream[q]*self.profile
        for m in xrange(self.size):
            n = round(self.tan*(m-self.size/2) + self.size/2)
            if abs(self.m+m) < self.grid.M and abs(self.n+n) < self.grid.N:
                F[self.m+m, self.n+n, 2] = vect[m]
                
    def Fy_hard(self, q, F):
        vect = self.amp*np.sin(2*np.pi*q/self.period + self.phi)*self.stream[q]*self.profile
        for n in xrange(self.size):
            m = round(self.tan*(n-self.size/2) + self.size/2)
            if abs(self.m+m) < self.grid.M and abs(self.n+n) < self.grid.N:
                F[self.m+m, self.n+n, 2] = vect[n]
    def loc_x(self):
        x = []
        y = []
        for m in xrange(self.size):
            n = round(self.tan*(m-self.size/2) + self.size/2)
            x.append(self.m+m)
            y.append(self.n+n)
        return x, y
    def loc_y(self):
        x = []
        y = []
        for n in xrange(self.size):
            m = round(self.tan*(n-self.size/2) + self.size/2)
            x.append(self.m+m)
            y.append(self.n+n)
        return x, y
        
