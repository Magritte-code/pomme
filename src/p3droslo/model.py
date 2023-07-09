import h5py
import torch
import numpy             as np
import matplotlib.pyplot as plt

from p3droslo.utils import interpolate


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
    reserved_keys = ["shape", "sizes", "free"]
    
    def __init__(self, sizes, shape, keys=[], dtau_warning_threshold=0.5):
        """
        Initialise a Tensor model.
        """
        # Check the keys
        for key in keys:
            if key in TensorModel.reserved_keys:
                raise ValueError(f"{key} is a reserved key word, please choose a different one.")
        # Set sizes of the model box
        self.sizes = make_object_with_len(sizes)
        # Set shape of Tensor variables
        self.shape = tuple(make_object_with_len(shape))
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
        # Set dtau_warning_threshold
        self.dtau_warning_threshold = dtau_warning_threshold
    
    
    def __getitem__(self, key):
        """
        Getter for variables (vars). Allows the use of [] operators.
        """
        return self.vars[key]
    
    
    def __setitem__(self, key, value):
        """
        Setter for variables (vars). Allows the use of [] operators.
        """
        if key in TensorModel.reserved_keys:
            raise ValueError(f"{key} is a reserved key word, please choose a different one.")
        if isinstance(value, torch.Tensor):
            self.vars[key] = value
        elif isinstance(value, np.ndarray):
            self.vars[key] = torch.from_numpy(value)
        else:
            raise ValueError("Only torch.Tensor is supported. (NumPy arrays are automatically cast.)")
        

    def parameters(self):
        """
        Return a list of all variables in the TensorModel.
        """
        return list(self.vars.values())
    
    
    def free_parameters(self):
        """
        Return a list of all variables in the TensorModel.
        """
        return [v for v in self.vars.values() if v.requires_grad]
    
    
    def dx(self, i):
        """
        Return the size of a model element.
        """
        return self.sizes[i] / self.shape[i]
    
    
    def integrate(self, var, axis=0):
        """
        Integrate a variable along an axis of the model box.
        """
        return self.dx(axis) * torch.cumsum(var, dim=axis)
    
    
    def integrate_out(self, var, axis=0):
        """
        Integrate a variable out along an axis of the model box.
        """
        return self.dx(axis) * torch.sum(var, dim=axis)

    
    def create_image(self, eta, chi, axis=0):
        """
        Formal solution of the transfer equation (discretised as TensorModel)
        """
        # Check for numercial issues.
        dtau_max = chi.max() * self.dx(axis)
        if dtau_max > self.dtau_warning_threshold:
            print('WARNING:')
            print(f'  dtau_max > {self.dtau_warning_threshold}, which might lead to numerical difficulties (dtau_max = {dtau_max})!')
        if chi.min() < 0.0:
            print('WARNING:')
            print(f'  Negative opacities encountered, which might lead to numerical difficulties!')
        # Compute the optical depth and the emerging intensity.
        tau = self.integrate    (chi,                 axis=axis)
        img = self.integrate_out(eta*torch.exp(-tau), axis=axis)
        return img
    

    def diff(self, arr, axis=-1):
        """
        Derivative along an axis.
        """
        # Prepend the one but first and append the one but last.
        # (Such that the first and last difference are the same.)
        ap0 = arr.index_select(dim=axis, index=torch.tensor([0]))
        ap1 = arr.index_select(dim=axis, index=torch.tensor([1]))
        am1 = arr.index_select(dim=axis, index=torch.tensor([arr.size(axis)-1]))
        am2 = arr.index_select(dim=axis, index=torch.tensor([arr.size(axis)-2]))
        pre = 2.0 * ap0 - ap1
        app = 2.0 * am1 - am2
        # Return the second-order difference.
        return 0.5 * (self.shape[axis] / self.sizes[axis]) * ( arr.diff(dim=axis, prepend=pre) + arr.diff(dim=axis, append=app) )


    def diff_x(self, arr):
        """
        Derivtive along x axis.
        """
        return self.diff(arr, axis=0)
    

    def diff_y(self, arr):
        """
        Derivtive along y axis.
        """
        return self.diff(arr, axis=1)

    
    def diff_z(self, arr):
        """
        Derivtive along z axis.
        """
        return self.diff(arr, axis=2)

    
    def keys(self):
        """
        Return the variable keys.
        """
        return self.vars.keys()
    
    
    def get_coords(self, origin=None):
        """
        Getter for the coordinates of each tensor location.
        
        Parameters
        ----------
        origin: array_like
            indices of the origin of the coordinate system (can be float).
            The dimension of the origin should match the dimension of the model.
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
            self.coords = np.moveaxis(np.indices(self.shape), 0, -1)
        # Shift coordinates w.r.t. origin
        self.coords = self.coords - self.origin
        # Scale coordinates to box size
        self.coords = np.moveaxis(self.coords * (np.array(self.sizes)/np.array(self.shape)), -1, 0)
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
            return np.abs(self.coords)
        else:
            return np.linalg.norm(self.coords, axis=0)
        

    def get_radial_direction(self, origin=None):
        """
        Getter for the radial direction of the model.
        """    
        coords = self.get_coords(origin)
        radius = self.get_radius(origin)

        direction = (1.0 / radius) * coords
        
        return torch.from_numpy(direction)
        
        
    def apply(self, func, exclude=[], include=[]):
        """
        Apply the given functional to all model variables.
        """
        res = torch.zeros(1)
        for key in self.keys():
            if key not in exclude:
                res += func(self[key])
        for var in include:
            res += func(var)
        return res

    
    def apply_to_fields(self, func, exclude=[], include=[]):
        """
        Apply the given functional to all model fields.
        """
        res = torch.zeros(1)
        for key in self.keys():
            if self.is_field(key) and (key not in exclude):
                res += func(self[key])
        for var in include:
            res += func(var)
        return res
    
    
    def free(self, keys):
        """
        Indicates which variables can freely be adjusted in optimistation.
        (and hence require a gradient.)
        """
        # Make sure that keys are iterable
        if not isinstance(keys, list):
            keys = [keys]
        # Free variables 
        for key in keys:
            self.vars[key].requires_grad_(True)
            
            
    def free_all(self):
        """
        Makes all model variables free.
        """
        self.free(list(self.keys()))


    def fix(self, keys):
        """
        Indicates which variables are fixed in optimistation.
        (and hence do not require a gradient.)
        """
        # Make sure that keys are a list
        if not isinstance(keys, list):
            keys = [keys]
        # Free variables 
        for key in keys:
            self.vars[key].requires_grad_(False)
            
            
    def is_field(self, key):
        """
        Check if the key corresponds to a 1D variable 
        """
        return self[key].shape == self.shape
        
        
    def get_detached_clone(self):
        """
        Returns a copy of the model, detached from this model. 
        """
        # Create a new TensorModel
        model = TensorModel(sizes=self.sizes, shape=self.shape)
        # Add detached clones of all variables
        for key in self.keys():
            model[key] = self[key].detach().clone()
        # Return the new model
        return model
    
    
    def info(self):
        """
        Print info about the TensorModel.
        """
        print("Variable key:              Free/Fixed:   Field:    Min:           Mean:          Max:")
        for key in self.keys():
            var = self[key]
            if var.requires_grad:
                fof = "Free"
            else:
                fof = "Fixed"
            if self.is_field(key):
                field = "True"
            else:
                field = "False"
            print(f"  {key:<25}  {fof:<5}         {field:<5}    {var.min():+.3e}     {var.mean():+.3e}     {var.max():+.3e}")
        print("sizes:", self.sizes)
        print("shape:", self.shape)

        
    @staticmethod
    def print_diff(model_1, model_2):
        print("Variable:                 Min:          Mean:         Max:")
        for key in model_1.keys():
            diff = torch.abs(model_1[key] - model_2[key])
        print(f"  {key:<25} {diff.min():.3e}     {diff.mean():.3e}     {diff.max():.3e}")


    def save(self, fname):
        """
        Save the TensorModel to disk as an HDF5 file.
        """
        def overwrite_dataset(file, key, data):
            try:
                del file[key]
            except:
                pass
            file[key] = data
        # Open or create an HDF5 file
        with h5py.File(fname, "a") as file:
            # Store the shape and sizes
            overwrite_dataset(file, "shape", self.shape)
            overwrite_dataset(file, "sizes", self.sizes)
            # Create a list to store the keys of the free variables
            free = []
            # Store the variables
            for key in self.keys():
                overwrite_dataset(file, key, self[key].data)
                # Check if it is a free variables
                if self[key].requires_grad:
                    # If so, store its key
                    free.append(key)
            # Store the keys of the free variables
            overwrite_dataset(file, "free", free)
        
        
    @staticmethod
    def load(fname):
        """
        Load a TensorModel from an HDF5 file.
        """
        with h5py.File(fname, "r") as file:
            # Read the shape and sizes
            shape = np.array(file["shape"])
            sizes = np.array(file["sizes"])
            # Create a TensorModel object
            model = TensorModel(shape=shape, sizes=sizes)
            # Read the list of keys of the free variables
            free = [f.decode("utf-8") for f in np.array(file["free"])]
            # Read the variables
            for key in file.keys():
                if key not in TensorModel.reserved_keys:
                    model[key] = np.array(file[key])
                    if key in free:
                        model[key].requires_grad_(True)
        # Return loaded model
        return model

    
    def interpolate_from_model(self, model, detach=True):
        """
        Interpolate the varialbles of another model into this model.
        """
        for key in model.keys():
            if model.is_field(key):
                self[key] = interpolate(model[key], self.shape).clone()
            else:
                self[key] = model[key].clone()
            # If required, detach tensor from the compute graph
            if detach:
                self.vars[key].detach_()
        


class SphericallySymmetric():
    """
    Spherically Symmetric model functionality.
    """
    def __init__(self, model_1D):
        """
        Constructor for a spherically symmetric model from a 1D TensorModel.
        """
        # Check if the input model is in fact 1D
        if model_1D.dimension != 1:
            raise ValueError(f"Input model is not 1D. model_1D.dimension = {model_1D.dimension}.")

        # Define the underlying 1D and 2D models
        self.model_1D = model_1D
        self.model_2D = TensorModel(
            sizes=(2*self.model_1D.sizes[0], self.model_1D.sizes[0]),
            shape=(2*self.model_1D.shape[0], self.model_1D.shape[0])
        )
        
        # Set origin for the 2D model halfway along the x axis
        origin_2D = np.array([0.5*(self.model_2D.shape[0]-1), 0.0])

        # Get linear interpolation utilities
        self.i_min, self.i_max, self.l = self.linint_utils(self.model_2D, origin_2D)
    
        # Get the circular weights
        self.c_weights = self.get_circular_weights()


    def linint_utils(self, model, origin):
        """
        Getter for the linear interpolation utilities.
        """
        # Compute the radii in the 2D model
        rs = torch.from_numpy(model.get_radius(origin=origin))

        # Compute the 2D radii in the 1D model
        rs = (self.model_1D.shape[0] / self.model_1D.sizes[0]) * rs

        # Compute the corresponding indices of the 2D radii in the 1D model
        i_min = torch.floor(rs).type(torch.int64)
        i_max = i_min + 1

        # Bound the indices from below
        i_min[i_min < 0] = 0
        i_max[i_max < 0] = 0

        # Bound the indices form above
        i_min[i_min >= self.model_1D.shape[0]] = self.model_1D.shape[0]-1
        i_max[i_max >= self.model_1D.shape[0]] = self.model_1D.shape[0]-1

        # Compute the scaling factor
        l = rs - i_min.type(torch.float64)
        
        return i_min, i_max, l
        
         
    def map_variables(self, keys):
        """
        Map a 1D Tensor Model variable to a 2D TensorModel variable assuming spherical symmetry.
        """    
        # Precompute the linear map
        a = self.l
        b = 1.0 - self.l
        # Map variables
        for key in keys:
            if self.model_1D.is_field(key):
                self.model_2D[key] = a*self.model_1D[key][self.i_max] + b*self.model_1D[key][self.i_min]
            else:
                self.model_2D[key] = self.model_1D[key]
 

    def map_1D_to_2D(self):
        """
        Map a 1D TensorModel to a 2D TensorModel assuming spherical symmetry.
        """    
        self.map_variables(self.model_1D.keys())

        
    def create_3D_model(self, sizes, shape, origin, detach=True):
        """
        Create a 3D TensorModel for the current spherically symmetric model, and given an origin.
        If detach=True, as is by default, all model tensors will be detached form the the spherical model.
        """
        # Check sizes
        if len(sizes) != 3:
            raise ValueError("Provided sizes do not correspond to a 3D model.")
        # Check shape
        if len(shape) != 3:
            raise ValueError("Provided shape does not correspond to a 3D model.")
        # Create a 3D tensor model
        model_3D = TensorModel(sizes=sizes, shape=shape)
        # Compute the linear integration utilities
        i_min, i_max, l = self.linint_utils(model_3D, origin)
        # Precompute the linear map
        a = l
        b = 1.0 - l
        # Map variables
        for key in self.model_1D.keys():
            if self.model_1D.is_field(key):
                model_3D[key] = a*self.model_1D[key][i_max] + b*self.model_1D[key][i_min]
            else:
                model_3D[key] = self.model_1D[key].clone()
            # If required, detach tensor from the compute graph
            if detach:
                model_3D[key].detach_()
        # Return the 3D TensorModel
        return model_3D

    
    def get_circular_weights(self):
        """
        Getter for the circular weights.
        """
        coords = self.model_2D.get_coords()
        
        return torch.from_numpy(2.0*np.pi*coords[1,0])
    
    
    def integrate_intensity(self, img):
        """
        Computes the integrated intensity.
        """
        return torch.einsum("if,i -> f", img, self.c_weights)
    

    def plot(self, keys=None, exclude=[]):
        """
        Plot the model parameters.
        """
        # Pick which variables to plot
        if keys is None:
            keys = self.model_1D.keys()
        # Get the radial coordinate of the model
        r = self.model_1D.get_radius()
        # Plot each 1D model variable
        for key in keys:
            if key not in exclude:
                if self.model_1D.is_field(key):
                    plt.figure(dpi=130)
                    plt.plot(r, self.model_1D[key].data)
                    plt.xlabel('r')
                    plt.ylabel(key)
                else:
                    print(f"{key:<21}{self.model_1D[key].data}")
        