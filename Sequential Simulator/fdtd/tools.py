###################
## FDTD SPECIFIC ##
###################

## Modules
import numpy as np
import scipy.sparse as sp
from constants import c

## Field Components
def x(F): # Take x component
    return F[..., 0]
def y(F): # Take y component
    return F[..., 1]
def z(F): # Take z component
    return F[..., 2]

## Simple functions
def sumc(F): # Sum of components
    return F[:,:,0]+F[:,:,1]+F[:,:,2]
def mag(F): # Magnitude of the vectorfield
    return np.sqrt(F[:,:,0]**2+F[:,:,1]**2+F[:,:,2]**2)

## Matrix multiply
# Used for matrix multiplication between a 3x3 matrix and a 3 component vector on an MxN grid.
from numpy.core.umath_tests import matrix_multiply as matrix_multiply0
def matrix_multiply(A,F):
    return matrix_multiply0(A,F[..., None])[...,0] #The [..., None] is necessary to convert the 3 component vector to a 3x1 matrix

## ASSYMETRIC DIFFERENCE FUNCTIONS
def ldx(F): # Low Difference of a field in the x direction
    ret = np.zeros(F.shape)
    ret[:-1,:] = F[1:,:]-F[:-1,:]
    ret[-1,:] = -F[-1,:]
    return ret
dEz_Ex = dEz_Hy = dEy_Hz = dHx_Hz = ldx # Used to extrapolate Ez to Ex pos. or Ey/Hx to Hz pos.

def ldy(F): # Low Difference of a field in the y direction 
    ret = np.zeros(F.shape)
    ret[:,:-1] = F[:,1:]-F[:,:-1]
    ret[:,-1] = -F[:,-1]
    return ret
dEz_Ey = dEz_Hx = dEx_Hz = dHy_Hz = ldy # Used to extrapolate Ez to Ey pos. or Ex/Hy to Hz pos.

def hdx(F): # High Difference of a field in the x direction
    ret = np.zeros(F.shape)
    ret[1:,:] = F[1:,:]-F[:-1,:]
    ret[0,:] = F[0,:]
    return ret
dHz_Hx = dHz_Ey = dHy_Ez = dEx_Ez = hdx # Used to extrapolate Hz to Hx pos. or Hy/Ex to Ez pos.

def hdy(F): # High Difference of a field in the y direction 
    ret = np.zeros(F.shape)
    ret[:,1:] = F[:,1:]-F[:,:-1]
    ret[:,0] = F[:,0]
    return ret
dHz_Hy = dHz_Ex = dHx_Ez = dEy_Ez = hdy # Used to extrapolate Hz to Hy pos. or Hx/Ey to Ez pos.

## SYMMETRIC DIFFERENCE FUNCTIONS
def sdx(F): # Symmetric Difference of a field in the x direction
    ret = np.zeros(F.shape)
    ret[1:-1,:] = 0.5*(F[2:,:]-F[:-2,:])
    ret[0,:] = 0.5*F[1,:]
    ret[-1,:] = -0.5*F[-2,:]
    return ret
def sdy(F): # Symmetric Difference of a field in the y direction
    ret = np.zeros(F.shape)
    ret[:,1:-1] = 0.5*(F[:,2:]-F[:,:-2])
    ret[:,0] = 0.5*F[:,1]
    ret[:,-1] = -0.5*F[:,-2]
    return ret

## ASYMMETRIC AVERAGE FUNCTIONS
def lax(F): # Low Average of a field in the x direction
    ret = np.zeros(F.shape)
    ret[:-1,:] = 0.5*(F[1:,:]+F[:-1,:])
    ret[-1,:] = 0.5*F[-1,:]
    return ret
Ez_Ex = Ez_Hy = Ey_Hz = Hx_Hz = lax # Used to extrapolate Ez to Ex pos. or Ey/Hx to Hz pos.

def lay(F): # Low Avarage of a field in the y direction
    ret = np.zeros(F.shape)
    ret[:,:-1] = 0.5*(F[:,1:]+F[:,:-1])
    ret[:,-1] = 0.5*F[:,-1]
    return ret
Ez_Ey = Ez_Hx = Ex_Hz = Hy_Hz = lay # Used to extrapolate Ez to Ey pos. or Ex/Hy to Hz pos.

def hax(F): # High Average of a field in the x direction
    ret = np.zeros(F.shape)
    ret[1:,:] = 0.5*(F[1:,:]+F[:-1,:])
    ret[0,:] = 0.5*F[0,:]
    return ret
Hz_Hx = Hz_Ey = Hy_Ez = Ex_Ez = hax # Used to extrapolate Hz to Hx pos. or Hy/Ex to Ez pos.

def hay(F): # High Avarage of a field in the y direction
    ret = np.zeros(F.shape)
    ret[:,1:] = 0.5*(F[:,1:]+F[:,:-1])
    ret[:,0] = 0.5*F[:,0]
    return ret
Hz_Hy = Hz_Ex = Hx_Ez = Ey_Ez = hay # Used to extrapolate Hz to Hy pos. or Hx/Ey to Ez pos.

## SYMMETRIC AVERAGE FUNCTIONS
def savx(F): # Symmetric Avarage of a field in the x direction
    ret = np.zeros(F.shape)
    ret[1:-1,:] = 0.5*(F[2:,:]+F[:-2,:])
    ret[0,:] = 0.5*(F[0,:]+F[1,:])
    ret[-1,:] = 0.5*(F[-2,:]+F[-1.:])
    return ret
def savy(F): # Symmetric Avarage of a field in the y direction
    ret = np.zeros(F.shape)
    ret[:,1:-1] = 0.5*(F[:,2:]+F[:,:-2])
    ret[:,0] = 0.5*(F[:,0]+F[:,1])
    ret[:,-1] = 0.5*(F[:,-2]+F[:,-1])
    return ret

## SPECIFIC E-FIELD FUNCTIONS
def curl_E(E): # Transforms an E field into an H field by performing a curl
    ret = np.zeros(E.shape)
    ret[:,:-1,0]  = E[:,1:,2]-E[:,:-1,2]
    ret[:,-1,0]   = -E[:,-1,2]
    ret[:-1,:,1]  = -E[1:,:,2]+E[:-1,:,2]
    ret[-1,:,1]   = E[-1,:,2]
    ret[:-1,:,2]  = E[1:,:,1]-E[:-1,:,1]
    ret[-1,:,2]   = -E[-1,:,1]
    ret[:,:-1,2] -= E[:,1:,0]-E[:,:-1,0]
    ret[:,-1,2]  -= -E[:,-1,0]
    return ret

def curl_E_3D( E ):
    ret = np.zeros(E.shape)
    
    # x - component
    ret[:,:-1,:,0] += (E[:,1:,:,2] - E[:,:-1,:,2])
    ret[:, -1,:,0] -=  E[:,-1,:,2]
    ret[:,:,:-1,0] -= (E[:,:,1:,1] - E[:,:,:-1,1])
    ret[:,:, -1,0] +=  E[:,:,-1,1]

    # y-component
    ret[:,:,:-1,1] += (E[:,:,1:,0] - E[:,:,:-1,0])
    ret[:,:, -1,1] -=  E[:,:,-1,0]
    ret[:-1,:,:,1] -= (E[1:,:,:,2] - E[:-1,:,:,2])
    ret[ -1,:,:,1] +=  E[-1,:,:,2]
    
    # z - component
    ret[:-1,:,:,2] += (E[1:,:,:,1] - E[:-1,:,:,1])
    ret[ -1,:,:,2] -=  E[-1,:,:,1]
    ret[:,:-1,:,2] -= (E[:,1:,:,0] - E[:,:-1,:,0])
    ret[:, -1,:,2] +=  E[:,-1,:,0]
    
    return ret[:,:,1,:]

def E_Ez(E): # Interpolates all the values to the Ez position.
    ret = E.copy()
    ret[...,0] = Ex_Ez(E[...,0])
    ret[...,1] = Ey_Ez(E[...,1])
    return ret

def E_Ex(E): # Interpolates all the values to the Ex position.
    ret = E.copy()
    ret[...,1] = Ez_Ex(Ey_Ez(E[...,1])) # Could be done better
    ret[...,2] = Ez_Ex(E[...,2])
E_Hy = E_Ex

def E_Ey(E): # Interpolates all the values to the Ey position.
    ret = E.copy()
    ret[...,0] = Ez_Ey(Ex_Ez(E[...,0])) # Could be done better
    ret[...,2] = Ez_Ey(E[...,2])
E_Hx = E_Ey

## SPECIFIC H-FIELD FUNCTIONS
def curl_H(H): # Transforms an H field into an E field by performing a curl
    ret = np.zeros(H.shape)
    ret[:,1:,0]  = H[:,1:,2]-H[:,:-1,2]
    ret[:,0,0]   = H[:,0,2]
    ret[1:,:,1]  = H[:-1,:,2]-H[1:,:,2]
    ret[0,:,1]   = -H[0,:,2]
    ret[1:,:,2]  = H[1:,:,1]-H[:-1,:,1]
    ret[0,:,2]   = H[0,:,1]
    ret[:,1:,2] -= H[:,1:,0]-H[:,:-1,0]
    ret[:,0,2]  -= H[:,0,0]
    return ret

def curl_H_3D( H ):
    ret = np.zeros(H.shape)
    
    # x - component
    ret[:,1:,:,0] += (H[:,1:,:,2] - H[:,:-1,:,2])
    ret[:,0 ,:,0] +=  H[:,0 ,:,2]
    ret[:,:,1:,0] -= (H[:,:,1:,1] - H[:,:,:-1,1])
    ret[:,:,0 ,0] -=  H[:,:,0 ,1]
    
    # y - component
    ret[:,:,1:,1] += (H[:,:,1:,0] - H[:,:,:-1,0])
    ret[:,:,0 ,1] +=  H[:,:,0 ,0]
    ret[1:,:,:,1] -= (H[1:,:,:,2] - H[:-1,:,:,2])
    ret[0 ,:,:,1] -=  H[0,:,:,2]
    
    # z - component
    ret[1:,:,:,2] += (H[1:,:,:,1] - H[:-1,:,:,1])
    ret[0 ,:,:,2] +=  H[0 ,:,:,1]
    ret[:,1:,:,2] -= (H[:,1:,:,0] - H[:,:-1,:,0])
    ret[:,0 ,:,2] -=  H[:,0 ,:,0]
    
    return ret[:,:,1,:]

def H_Hz(H): # Interpolates all the values to the Hz position.
    ret = H.copy()
    ret[...,0] = Hx_Hz(H[...,0])
    ret[...,1] = Hy_Hz(H[...,1])
    return ret

def H_Hx(H): # Interpolates all the values to the Hz position.
    ret = H.copy()
    ret[...,1] = Hz_Hx(Hy_Hz(H[...,1])) # Could be done better
    ret[...,2] = Hz_Hx(H[...,2])
    return ret
H_Ey = H_Hx

def H_Hy(H): # Interpolates all the values to the Hz position.
    ret = H.copy()
    ret[...,0] = Hz_Hy(Hx_Hz(H[...,0])) # Could be done better
    ret[...,2] = Hz_Hy(H[...,2])
    return ret
H_Ex = H_Hy

def H_Ez(H): # Interpolates all the values to the Ez position.
    ret = H.copy()
    ret[...,0] = Hx_Ez(H[...,0])
    ret[...,1] = Hy_Ez(H[...,1])
    ret[...,2] = 0.5*Hx_Ez(Hz_Hx(H[...,2]))+0.5*Hy_Ez(Hz_Hy(H[...,2]))
    return ret

## POYNTING VECTOR
def poynting(E,H):
    Ex,Ey,Ez = E_Ez(E).transpose(2,0,1)
    Hx,Hy,Hz = H_Ez(H).transpose(2,0,1)
    return c*np.array([ (Ey*Hz-Ez*Hy) , (Ez*Hx-Ex*Hy) , (Ex*Hy-Ey*Hx) ]).transpose(1,2,0)

def irradiance(E,H):
    Ex,Ey,Ez = E_Ez(E).transpose(2,0,1)
    Hx,Hy,Hz = H_Ez(H).transpose(2,0,1)
    return c*np.sqrt((Ey*Hz-Ez*Hy)**2 + (Ez*Hx-Ex*Hy)**2 + (Ex*Hy-Ey*Hx)**2)
    

## ELECTRO OPTIC TENSORS
def inv_epsilon(S, eps, r): # Inverse of the permittivity tensor [constructed from pockels tensor and electric field]
    epsX,epsY,epsZ = eps
    epsilonMatrix = np.zeros((S[...,0].shape[0], S[...,1].shape[1], 3, 3))   
    epsilonMatrix[...,0,0] = 1./epsX + r[0,0]*S[...,0] + r[0,1]*S[...,1] + r[0,2]*S[...,2]
    epsilonMatrix[...,1,1] = 1./epsY + r[1,0]*S[...,0] + r[1,1]*S[...,1] + r[1,2]*S[...,2]
    epsilonMatrix[...,2,2] = 1./epsZ + r[2,0]*S[...,0] + r[2,1]*S[...,1] + r[2,2]*S[...,2]
    epsilonMatrix[...,1,2] = epsilonMatrix[...,2,1] = + r[3,0]*S[...,0] + r[3,1]*S[...,1] + r[3,2]*S[...,2]
    epsilonMatrix[...,0,2] = epsilonMatrix[...,2,0] = + r[4,0]*S[...,0] + r[4,1]*S[...,1] + r[4,2]*S[...,2]
    epsilonMatrix[...,0,1] = epsilonMatrix[...,1,0] = + r[5,0]*S[...,0] + r[5,1]*S[...,1] + r[5,2]*S[...,2]
    return epsilonMatrix


## Airy Function
from scipy.special import airy as spairy
def airy(x, a=0.1):
    y,_,_,_ = spairy(x)
    return y*np.exp(a*x)

## Required refractive index for a certain reflectivity
def index(object_index, reflectivity):
    return object_index*(1+reflectivity+2*np.sqrt(reflectivity))/(1-reflectivity)
    
## Linear algebra Matrix Solvers
from scipy.sparse import linalg
def solve(A, b, **kwargs):
    return (linalg.spsolve(A,b), )
bicgstab = linalg.bicgstab
lgmres = linalg.lgmres

## Creates a Sparse matrix implementing curl and div equations for solving maxwell's equations to find the space charge field
def sparse_matrix(M, N):
    '''
    FAST VERSION
    Makes a sparse matrix to solve the system Ax=b.
    '''
    a = np.diag(np.ones(N-1, dtype = int))
    a = np.concatenate((a, np.zeros((1,N-1), dtype = int)), 0)
    
    a[1:,:] = a[1:,:] - a[:-1,:]
    
    divY = a[None, :, :]
    divY = divY.repeat(M, 0)
    divY = sp.block_diag(divY, format='csr')
    curlX = np.transpose(a)
    curlX = curlX[None, :, :]
    curlX = curlX.repeat(M-1, 0)
    curlX = sp.block_diag(curlX, format="csr")
    
    a = np.diag(np.ones(N, dtype = int))
    b = a[None, :, :]
    b = b.repeat(M-1, 0)
    c = sp.block_diag(b, format = "csr")
    l = sp.csr_matrix((N, (M-1)*N), dtype = int)
    divX = sp.vstack((c, l), format='csr') - sp.vstack((l, c), format='csr')

    a = np.diag(np.ones(N-1, dtype = int))
    b = a[None, :, :]
    b = b.repeat(M-1, 0)
    c = sp.block_diag(b, format="csr")
    l = sp.csr_matrix(((N-1)*(M-1), N-1), dtype = int)
    curlY = -sp.hstack((c, l), format='csr') + sp.hstack((l, c), format='csr')    
    
    # Div(E) = q
    uphalf = sp.hstack((divX, divY), format='csr')
    # Curl(E) = 0
    downhalf = sp.hstack((curlX, curlY), format='csr')

    res = sp.vstack((uphalf, downhalf), format='csr')
    delete_row_csr(res, 0)
    
    return res.tocsc()

def delete_row_csr(mat, i): # Matrix casting is not allowed for csr matrices. Therefore we define this function to remove a row.
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])


############################
## OTHER USEFUL FUNCTIONS ##
############################
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


from warnings import catch_warnings as cw
def catch_warnings(record=True):
    return cw(record=record) 
    

# Get local time in Hours and Minutes
from time import localtime, strftime
def time(): 
    return strftime("%H:%M", localtime())

def log(s): # Log to file [log.txt]
    with open("log.txt", "a") as log:
        print >> log, s

import os
def cd(folder): # Go to folder. If folder does not exist, create folder.
    folder = str(folder)
    if folder != '..' and not os.path.isdir(folder):
        os.mkdir(folder)
    os.chdir(folder)
    
import sys
from contextlib import contextmanager
@contextmanager
def suppress_print(): # Suppress print output
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# Send an e-mail [for example with progress of the simulation]
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
def mail(msg='', txt='', subj='', imgs=[], to='floris.laporte@gmail.com'):
    fromaddr = 'simulationsfloris@gmail.com'
    password = 'hallo.floris'
    
    if txt:
        msg = open(txt,'r').read()
    
    m = MIMEMultipart()
    m['Subject'] = subj
    m.attach(MIMEText(msg))
    
    if type(imgs)==str:
        imgs = [imgs]
    for img in imgs:
        im = open(img+'.png', 'rb').read()
        m.attach(MIMEImage(im,name='img'))
    try: # Communication
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        server.login(fromaddr,password)
        server.sendmail(fromaddr, to, m.as_string())
        server.quit()
    except:
        print "Mail not send."


















    
