""" A BASIC FDTD MODULE """

__author__ = "Floris Laporte"
__copyright__ = "Copyright (c) 2016"
__email__ = "floris.laporte@gmail.com"

#############
## MODULES ##
#############

__all__ = ['constants',
           'crystal',
           'detector',
           'grid',
           'plotter',
           'pml',
           'source',
           'tools']

import sys
sys.dont_write_bytecode = True # Can be removed
