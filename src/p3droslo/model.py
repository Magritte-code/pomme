import torch
import numpy as np


def make_object_with_len(obj):
    """
    Turn object into a tuple if it does not have a __len__ attribute.
    """
    if hasattr(obj, "__len__"):
        return obj
    else:
        return (obj,)
        

class TensorModel():
    """
    A (deterministic) model in which every variable is represented by a 3-tensor.
    """
    
    def __init__(self, sizes, shape, keys=[]):
        """
        Initialise a Tensor model.
        """
        # Set sizes of the model box
        self.sizes = make_object_with_len(sizes)
        # Set shape of Tensor variables
        self.shape = make_object_with_len(shape)
        # Set the dimension of the model
        if len(self.shape) == len(self.sizes):
            self.dimension = len(self.shape)
        else:
            raise ValueError(f"sizes and shape have a different number of dimensions: {len(self.sizes)} and {len(self.shape)}")
        # Initialise variables with random Tensor with appropriate shape 
        self.vars = {v:torch.rand(self.shape, requires_grad=True, dtype=torch.float64) for v in keys}
        # Extract geometrical properties
        self.origin = None
        self.coords = None
    
    
    def __getitem__(self, key):
        """
        Getter for variables (vars). Allows the use of [] operators.
        """
        return self.vars[key]
    
    
    def __setitem__(self, key, value):
        """
        Setter for variables (vars). Allows the use of [] operators.
        """
        self.vars[key] = value
    
    
    def parameters(self):
        """
        Return a list of all variables in the TensorModel.
        """
        return list(self.vars.values())
    
    
    def dx(self, i):
        """
        Return the size of a model element.
        """
        return self.sizes[i] / self.shape[i]
    
    
    def integrate(self, var, axis=0):
        """
        Integrate a variable along an axis of the model box.
        """
        return torch.cumsum(self.dx(axis)*var, dim=axis)
    
    
    def integrate_out(self, var, axis=0):
        """
        Integrate a variable out along an axis of the model box.
        """
        return torch.sum(self.dx(axis)*var, dim=axis)

    
    def create_image(self, eta, chi, axis=0):
        """
        Formal solution of the transfer equation (discretised as TensorModel)
        """
        tau = self.integrate    (chi,                 axis=axis)
        img = self.integrate_out(eta*torch.exp(-tau), axis=axis)
        return img

    
    def keys(self):
        """
        Return the variable keys.
        """
        return self.vars.keys()
    
    
    def get_coords(self, origin=None):
        """
        Getter for the coordinates of each tensor location.
        """
        # Cast origin into a numpy array
        if origin is not None:
            self.origin = np.array(origin)
        elif self.origin is None:
            self.origin = np.zeros(self.dimension)
        # Compute the coordinates of each tensor location
        if self.dimension == 1:
            self.coords = np.arange(self.shape[0])
        else:
            self.coords = np.indices(self.shape).T
        # Shift coordinates w.r.t. origin
        self.coords = self.coords - self.origin.T
        # Scale coordinates to box size
        self.coords = (self.coords * (np.array(self.sizes)/np.array(self.shape)).T).T
        return self.coords

    
    def get_radius(self, origin=None):
        """
        Getter for the radial cooridnate of each location.
        """
        # Check if coords are already set
        if (self.coords is None) or not np.all(origin == self.origin):
            self.coords = self.get_coords(origin)
        # In 1D there only is a radial coord
        if self.dimension == 1:
            return self.coords
        else:
            return np.linalg.norm(self.coords, axis=0)
        
        
    def apply(self, func):
        """
        Apply the given functional to tha model variables.
        """
        res = torch.zeros(1)
        for par in self.parameters():
            res += func(par)
        return res