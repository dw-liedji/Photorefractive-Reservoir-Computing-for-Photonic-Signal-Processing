''' Optional wrapper for matplotlib.pyplot with some style changes'''

#############
## MODULES ##
#############
import numpy as np
import datetime
import os
import matplotlib as mpl

working_on_server = '/home/photonics/' in os.path.abspath(__file__)
if working_on_server: mpl.use('Agg') # Server does not support the standard mpl framework

import matplotlib.pyplot as plt
from matplotlib.pyplot import *

import seaborn as sns # Seaborn is not installed on the server


###########
## STYLE ##
###########
default_cmap='RdBu'
rcParams['legend.fontsize'] = 'medium'
rcParams['image.cmap'] = default_cmap # Default colormap
#rcParams['axes.grid'] = False # Remove grid introduced by seaborn
#rcParams['axes.facecolor'] = 'white'
rcParams['axes.ymargin'] = 0.1
rcParams['axes.xmargin'] = 0

rcParams['lines.linewidth'] = 2.5

rcParams['xtick.direction'] = 'inout'
rcParams['ytick.direction'] = 'inout'

#########
## NEW ##
#########
# Global Font change
def set_font(tex = True, serif = True):
    rcParams['text.usetex'] = tex            
    if serif:
        rcParams['font.family'] = 'serif'
    elif tex:
        rc('text.latex', preamble=r'\usepackage{cmbright}')
    else:
        rcParams['font.family'] = 'sans-serif'

###############
## REDEFINED ##
###############

def savefig(name=None):
    if name is None:
        plt.savefig(str(datetime.datetime.now().date()).replace('-','')+'-'+str(datetime.datetime.now().time()).split('.')[0].replace(':',''))
    else:
        plt.savefig(name)
    plt.clf()
    plt.close()        
  
def show2():
    pass
show = show2 if working_on_server else plt.show

def title(*args, **kwargs):
    if not 'fontsize' in kwargs:
        kwargs['fontsize'] = 18
    return plt.suptitle(*args, **kwargs)

def figlegend(handles=None, labels=None, loc=None, **kwargs):
    if handles is None:
        axs = list(np.array(plt.gcf().get_axes(), dtype='object').flatten())
        handles = []
        labels2 = []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles = handles + h
            labels2 = labels2 + l
    if labels is None: labels = labels2
    if loc is None: loc = "lower center"
    return plt.figlegend(handles, labels, loc, **kwargs)

from mpl_toolkits.axes_grid1 import make_axes_locatable

def imshow(plots, **kwargs): # Imshow wrapper with extended functionality
    plots = np.array(plots)
    plots = np.atleast_2d(plots) if plots.dtype == object else np.array([[plots]])

    m, n = (plots.shape[0], plots.shape[1])
    
    fig, axs = plt.subplots(m,n)
    if m==1 and n==1:
        axs = np.array([[axs]], dtype='object')
    elif m==1:
        axs = np.array(axs, dtype='object')[None,:]
    elif n==1:
        axs = np.array(axs, dtype='object')[:,None]
    else:
        axs = np.array(axs, dtype='object')
        
    maxs = []
    mins = []
    for i in xrange(m):
        for j in xrange(n):
            maxs.append(plots[i,j].max())
            mins.append(plots[i,j].min())
    
    show_ticks = True if not "show_ticks" in kwargs else kwargs["show_ticks"] # Show ticks
    colorbar = "same" if not "colorbar" in kwargs else kwargs["colorbar"] # Show colorbar
    cmap = default_cmap if not "cmap" in kwargs else kwargs["cmap"] # Specify colormap
    lim = [min(mins), max(maxs)] if not "lim" in kwargs else kwargs["lim"] # Limits of the colorbar
    symm = False if not "symm" in kwargs else kwargs["symm"] # Symmetric limits of the colorbar
        
    imgs = np.zeros((m,n), dtype='object')
    for i in xrange(m):
        for j in xrange(n):
            ax =  axs[i,j]
            p = plots[i,j]
            
            if colorbar == "same":
                dummy = np.zeros(p.shape)
                dummy[0,0] = lim[0]
                dummy[0,1] = lim[1]
                dummy[1,0] =-lim[0] if symm else 0.
                dummy[1,1] =-lim[1] if symm else 0.            
                img = ax.imshow(dummy, cmap=cmap)
                img.set_data(p)
            else:
                img = ax.imshow(p, cmap=cmap)
            
            if colorbar:
                div = make_axes_locatable(ax)
                cax = div.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(img, cax=cax)
                plt.sca(ax)
                plt.sci(img)
                
            ax.xaxis.set_visible(show_ticks)
            ax.yaxis.set_visible(show_ticks)
            imgs[i,j] = img
    if imgs.shape[0] == 1 and imgs.shape[1] == 1:
        return imgs[0][0]
    if imgs.shape[0] == 1 or imgs.shape[1] == 1:
        return np.array(imgs).flatten()
    return imgs

from matplotlib import gridspec
def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, lim = [None, None], subplot_kw=None,
             gridspec_kw=None, show_x_axis = True, show_y_axis = True, return_gs = False, figlegend = 0.25, **fig_kw):
    """
    ADAPTED FROM pyplot.subplots
    """
    nplots = nrows*ncols    
    
    # for backwards compatibility
    sharex = "all" if type(sharex) is bool and sharex else "none"
    sharey = "all" if type(sharey) is bool and sharey else "none"
    share_values = ["all", "row", "col", "none"]
    if sharex not in share_values: raise ValueError("sharex [%s] must be one of %s" % (sharex, share_values))
    if sharey not in share_values: raise ValueError("sharey [%s] must be one of %s" % (sharey, share_values))
    if subplot_kw is None: subplot_kw = {}
    if gridspec_kw is None: gridspec_kw = {}

    fig = figure(**fig_kw)
    
    gs = gridspec.GridSpec(nrows, ncols, **gridspec_kw)
    if figlegend:
        gridspec_kw['height_ratios'] = gridspec_kw['height_ratios'] + [figlegend] if 'height_ratios' in gridspec_kw else [1]*nrows + [figlegend]
        gs = gridspec.GridSpec(nrows + 1, ncols, **gridspec_kw)

    # Create empty object array to hold all axes.  It's easiest to make it 1-d.
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    axarr[0] = fig.add_subplot(gs[0, 0], **subplot_kw)

    r, c = np.mgrid[:nrows, :ncols]
    r = r.flatten() * ncols
    c = c.flatten()
    lookup = {
        "none": np.arange(nplots),
        "all": np.zeros(nplots, dtype=int),
        "row": r,
        "col": c,
    }
    sxs = lookup[sharex]
    sys = lookup[sharey]
    

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        if sxs[i] == i:
            subplot_kw['sharex'] = None
        else:
            subplot_kw['sharex'] = axarr[sxs[i]]
        if sys[i] == i:
            subplot_kw['sharey'] = None
        else:
            subplot_kw['sharey'] = axarr[sys[i]]
        axarr[i] = fig.add_subplot(gs[i // ncols, i % ncols], **subplot_kw)

    # returned axis array will be always 2-d, even if nrows=ncols=1
    axarr = axarr.reshape(nrows, ncols)

    # turn off redundant tick labeling
    if sharex in ["col", "all"] and nrows > 1 or not show_x_axis:
        # turn off all but the bottom row
        for ax in axarr[:(-1 if show_x_axis else None), :].flat:
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)

    if sharey in ["row", "all"] and ncols > 1 or not show_y_axis:
        # turn off all but the first column
        for ax in axarr[:, (1 if show_y_axis else 0):].flat:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

    for row in axarr:
        for a in row:
            if not lim[0] is None: a.set_xlim(*lim[0])
            if not lim[1] is None: a.set_ylim(*lim[1])

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            ret = [fig, axarr[0, 0], gs]
        else:
            ret = [fig, axarr.squeeze(), gs]
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        ret = [fig, axarr.reshape(nrows, ncols), gs]
        
    if not return_gs: del ret[-1]   

    return ret
