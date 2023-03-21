import torch


class TensorModel():
    """
    A (deterministic) model in which every variable is represented by a 3-tensor.
    """
    
    def __init__(self, var_keys, box_sizes, box_shape):
        """
        Initialise a Tensor model.
        """
        # Set sizes of the model box.
        self.box_sizes = box_sizes
        # Set shape of Tensor variables. 
        self.box_shape = box_shape
        # Initialise variables with random Tensor with appropriate shape 
        self.vars = {v:torch.rand(self.box_shape, requires_grad=True, dtype=torch.float64) for v in var_keys}
        
    def __getitem__(self, var_key):
        """
        Getter for variables (vars). Allows the use of [] operators.
        """
        return self.vars[var_key]
    
    def __setitem__(self, var_key, value):
        """
        Setter for variables (vars). Allows the use of [] operators.
        """
        self.vars[var_key] = value
        
    def parameters(self):
        """
        Return a list of all variables in the TensorModel.
        """
        return list(self.vars.values())
    
    def dx(self, i):
        """
        Return the size of a model element.
        """
        return self.box_sizes[i] / self.box_shape[i]
    
    def integrate(self, var, axis=2):
        """
        Integrate a variable along an axis of the model box.
        """
        return torch.cumsum(self.dx(axis)*var, dim=axis)
    
    
    def integrate_out(self, var, axis=2):
        """
        Integrate a variable out along an axis of the model box.
        """
        return torch.sum(self.dx(axis)*var, dim=axis)

    def create_image(self, eta, chi, axis=2):
        """
        Formal solution of the transfer equation (discretised as TensorModel)
        """
        tau = self.integrate    (chi,                 axis=axis)
        img = self.integrate_out(eta*torch.exp(-tau), axis=axis)
        return img