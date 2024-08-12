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

        Parameters
        ----------
        keys : list
            List of keys for the losses.
        """
        # Store keys
        self.keys = keys
        # Provide entries for all losses, norm, and weight
        self.loss   = {key:torch.zeros(1) for key in self.keys}
        self.norm   = {key:1.0            for key in self.keys} 
        self.weight = {key:1.0            for key in self.keys} 
        self.losses = {key:[]             for key in self.keys}
        # Provide entries for tot loss
        self.loss_tot   = torch.zeros(1)
        self.losses_tot = []


    def reset(self):
        """
        Reset all losses and remove all stored losses.
        """
        # Provide entries for all losses, norm, and weight
        self.losses = {key:[] for key in self.keys}
        # Provide entries for tot loss
        self.loss_tot   = torch.zeros(1)
        self.losses_tot = []

    
    def __getitem__(self, key):
        """
        Getter for variables (vars). Allows the use of [] operators.

        Parameters
        ----------
        key : str
            Key of the variable to be returned.

        Returns
        -------
        torch.Tensor
            The requested variable.
        """
        return self.loss[key]
    
    
    def __setitem__(self, key, value):
        """
        Setter for variables (vars). Allows the use of [] operators.

        Parameters
        ----------
        key : str
            Key of the variable to be set.
        value : torch.Tensor
            Value to be set.

        Raises
        ------
        ValueError
            If the value is not a torch.Tensor.
        """
        if isinstance(value, torch.Tensor):
            # weight
            value *= self.weight[key]
            # normalise
            value *= self.norm[key]
            # store
            self.loss[key] = value
            self.losses[key].append(value.item())
        else:
            raise ValueError("Only torch.Tensor is supported.")


    def renormalise(self, key):
        """
        Reset the norm to one over the current loss.

        Parameters
        ----------
        key : str
            Key of the variable to be renormalised.
        """
        self.norm[key] *= 1.0 / self.loss[key].item()


    def renormalise_all(self):
        """
        Renormalise all losses.
        """
        for key in self.keys:
            self.renormalise(key)

            
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
        
        
def haar_loss_1D(arr):
    """
    Loss based on the Haar wavelet transform.
    """
    loss = torch.zeros(1)
    while len(arr) > 1:
        loss += nn.functional.mse_loss(arr[0::2], arr[1::2]) * len(arr)
        arr = 0.5*(arr[0::2] + arr[1::2])
    return loss


def fourier_loss_1D(arr):
    """
    Loss based on the (1D) Fourier transform.
    """
    # Take the (real) Fast Fourier Transform
    fft = torch.abs(torch.fft.rfft(arr))
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
    This is quantified as the variacne of the data in each radial bin.
    """
    def __init__(self, model, origin='centre', weights=None):
        """
        Constructor for the spherical loss.

        Parameters
        ----------
        model : TensorModel
            TensorModel object on which the loss should be applied.
        origin : array_like or str
            Origin of the spherical coordinates, given as indices of the origin
            of the coordinate system (can be float).
            The dimension of the origin should match the dimension of the model.
            If 'centre', the origin is set to the centre of the model.
        weights : torch.Tensor, optional
            Weights for the radial bins. If None, the weights are set to one.
        """
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
        """
        Evaluate the spherical loss.

        Parameters
        ----------
        var : torch.Tensor
            Variable for which the loss should be evaluated.

        Returns
        -------
        torch.Tensor
            The spherical loss for the given variable.
        """
        # Compute the standard deviation of the variable data in each radial bin.
        sph_std = torch.zeros(self.N)
        for i in range(self.N):
            sph_std[i] = var[self.masks[i]].std()
        return torch.mean(self.weights * sph_std)


# def steady_state_continuity_loss(model, rho, v_x, v_y, v_z):
#     """
#     Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives in Euler's equations.
#     """
#     # Continuity equation (steady state): div(ρ v) = 0
#     loss_cont = model.diff_x(rho * v_x) + model.diff_y(rho * v_y) + model.diff_z(rho * v_z)
#     # Squared average over the model
#     loss_cont = torch.mean((loss_cont / rho)**2)
#     return loss_cont