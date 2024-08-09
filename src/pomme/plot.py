import torch
import matplotlib.pyplot as plt

from ipywidgets import interact


def plot_cube_2D(cube):
    """
    Plot a slice along the third axis through a 3D cube.

    Parameters
    ----------
    cube : torch.Tensor
        3D cube to plot.
    
    Returns
    -------
    out : None
    """
    vmin = cube.min().item()
    vmax = cube.max().item()
    def plot(z):
        plt.figure(dpi=150)
        plt.imshow(cube[:,:,z].T.data, vmin=vmin, vmax=vmax, origin='lower')
    return interact(plot, z=(0,cube.shape[2]-1))

    
def plot_spectrum(cube):
    """
    Plot spectrum at a pixel for this observation.

    Parameters
    ----------
    cube : torch.Tensor
        3D cube to plot the spectrum of.
    
    Returns
    -------
    out : None
    """
    # Define a plot function
    ymin = cube.min().item()
    ymax = cube.max().item()
    def plot(i,j):
        plt.figure(dpi=150)
        plt.plot(cube[i,j,:].data)
        plt.ylim((ymin, ymax))

    # Return an interactive ipywidget
    return interact(plot, i=(0, cube.shape[0]-1), j=(0, cube.shape[1]-1))