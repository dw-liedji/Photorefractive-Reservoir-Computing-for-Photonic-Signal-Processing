#############
## MODULES ##
#############
import os
import numpy as np
import tools as tls

##########
## LINE ##
##########

## Parent class
class Detector():
    def __init__(self, grid): # Minimal initialization.
        self.grid = grid
        self.e = []
        self.h = []

        self.grid.detectors.append(self)
    @property
    def E(self):
        return np.array(self.e)
    @property
    def H(self):
        return np.array(self.h)
    
    def detect_E(self):
        self.e.append(self.grid.E[self.loc])
                
    def detect_H(self):
        self.h.append(self.grid.H[self.loc])
        
    def reset(self):
        del self.e[:]
        del self.h[:]

    def plot(self, ax=None):
        if ax is None:
            from matplotlib.pyplot import gca
            ax = gca()
        X,Y = self.loc
        x = np.arange(self.grid.N) if X == slice(None) else [X]*self.grid.N
        y = np.arange(self.grid.M) if Y == slice(None) else [Y]*self.grid.M
        ax.plot(x,y, color='cyan')


## Subclasses    
class HorizontalDetector(Detector):
    def __init__(self, grid, position):
        self.loc = (position, slice(None))
        Detector.__init__(self, grid)

class VerticalDetector(Detector):
    def __init__(self, grid, position):
        self.loc = (slice(None), position)
        Detector.__init__(self, grid)
