import torch
import torch.nn          as nn
import numpy             as np
import matplotlib.pyplot as plt


class Loss():
    """
    A convenience class to store losses.
    """
    
    def __init__(self, keys=[]):
        """
        Constructor for loss.
        """
        # Store keys
        self.keys = keys
        # Provide entries for all losses
        self.loss   = {key:torch.zeros(1) for key in self.keys}
        self.losses = {key:[]             for key in self.keys}
        # Provide entries for tot loss
        self.loss_tot   = torch.zeros(1)
        self.losses_tot = []
    
    
    def __getitem__(self, key):
        """
        Getter for variables (vars). Allows the use of [] operators.
        """
        return self.loss[key]
    
    
    def __setitem__(self, key, value):
        """
        Setter for variables (vars). Allows the use of [] operators.
        """
        if isinstance(value, torch.Tensor):
            self.loss[key] = value
            self.losses[key].append(value.item())
        else:
            raise ValueError("Only torch.Tensor is supported.")
            
            
    def tot(self):
        """
        Return the total loss.
        """
        # Initialise the total loss
        self.loss_tot = torch.zeros(1)
        # Add all losses
        for l in self.loss.values():
            self.loss_tot += l
        # Add tot list
        self.losses_tot.append(self.loss_tot.item())
        # Return the total loss
        return self.loss_tot
    
    
    def plot(self):
        """
        Plot the evolution of the losses.
        """
        plt.figure(dpi=130)
        # Plot the total loss
        plt.plot(self.losses_tot, label='tot', linestyle=':', c='k')
        # Plot all constituent losses
        for key in self.losses.keys():
            plt.plot(self.losses[key], label=key)
        plt.yscale('log')
        plt.legend()
        
        
def haar_loss_1D(a):
    loss = torch.zeros(1)
    while len(a) > 1:
        loss += nn.functional.mse_loss(a[0::2], a[1::2]) * len(a)
        a = 0.5*(a[0::2] + a[1::2])
    return loss


def fourier_loss_1D(a):
    """
    Loss based on the (1D) Fourier transform.
    """
    # Take the (real) Fast Fourier Transform
    fft = torch.abs(torch.fft.rfft(a))
    # Store the size of the resulting transform
    size = fft.size(0)
    # Define the weights
    wts = torch.arange(size)
    # Compute the mean square loss
    loss = torch.mean((fft*wts)**2)
    # Return the loss
    return loss


def diff_loss(arr):
    """
    Differential loss, quantifying the local change in a variable along the cartesian axes.
    """
    loss = torch.zeros(1)
    for d in range(arr.dim()):
        loss += torch.mean(torch.diff(arr, dim=d)**2)
    return loss


class SphericalLoss:
    """
    Copmutes the deviation from spherical symmetry. 
    """

    def __init__(self, model, origin, weights=None):
        # Get radial coordinates w.r.t. the given origin.
        r = model.get_radius(origin=origin)
        # Define the number of radial bins.
        self.N = int(np.mean(model.shape)//2)
        # Define the boundaries of the radial bins.
        r_min = r.min()
        r_max = r.max()
        # Extract the indices of the radial bins.
        self.r_ind = ((r - r_min) *  ((self.N - 1) / (r_max - r_min))).astype(np.int64)
        # Extract the spherical masks (which mask everything but radial bin 'i').
        self.masks = [self.r_ind==i for i in range(self.N)]
        # Set the weights, if necessary.
        if weights is None:
            self.weights = torch.ones(self.N)
        else:
            self.weights = weights

    def eval(self, var):
        # Compute the standard deviation of the variable data in each radial bin.
        sph_std = torch.zeros(self.N)
        for i in range(self.N):
            sph_std[i] = var[self.masks[i]].std()
        return torch.mean(self.weights * sph_std)