import h5py
import torch
import numpy             as np
import matplotlib.pyplot as plt

from pomme.utils import interpolate
from tqdm           import tqdm
from time           import perf_counter


def make_object_with_len(obj):
    """
    Turn object into a tuple if it does not have a __len__ attribute.

    Parameters
    ----------
    obj: object
        Object to be turned into a tuple if it does not have a __len__ attribute.

    Returns
    -------
    obj: object
        Object turned into a tuple if it does not already had a __len__ attribute.
    """
    if hasattr(obj, "__len__"):
        return obj
    else:
        return (obj,)
        

class TensorModel():
    """
    A (deterministic) model in which every variable is represented by a PyTorch tensor.
    """
    reserved_keys = ["shape", "sizes", "free"]
    
    def __init__(self, sizes, shape, keys=[], dtau_warning_threshold=0.5):
        """
        Constructor for a Tensor model object.

        Parameters
        ----------
        sizes: array_like
            The sizes of the model box.
        shape: array_like
            The shape of the Tensor variables.
        keys: list (optional)
            List of keys for the model variables.
        dtau_warning_threshold: float (optional)
            Threshold for the optical depth at which a warning is issued.

        Raises
        ------
        ValueError
            If the sizes and shape have a different number of dimensions.
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


    def deepcopy(self):
        """
        Create a deep copy of the TensorModel.
        """
        # Create a new TensorModel
        m = TensorModel(sizes=self.sizes, shape=self.shape, dtau_warning_threshold=self.dtau_warning_threshold)
        # Deepcopy all variables
        for key in self.keys():
            m[key] = self[key].detach().clone()
        # Return the new model
        return m


    def dim(self):
        """
        Getter for the dimension of the TensorModel.
        """
        return len(self.shape)
    

    def __getitem__(self, key):
        """
        Getter for variables (vars). Allows the use of [] operators.

        Parameters
        ----------
        key: str
            Key of the variable to be returned.

        Returns
        -------
        torch.Tensor
            The requested variable.
        """
        return self.vars[key]
    
    
    def __setitem__(self, key, value):
        """
        Setter for variables (vars). Allows the use of [] operators.

        Parameters
        ----------
        key: str
            Key of the variable to be set.
        value: torch.Tensor
            The value to be set.
        
        Raises
        ------
        ValueError
            If the key is a reserved key word.
        """
        if key in TensorModel.reserved_keys:
            raise ValueError(f"{key} is a reserved key word, please choose a different one.")
        if isinstance(value, torch.Tensor):
            self.vars[key] = value
        elif isinstance(value, np.ndarray):
            self.vars[key] = torch.from_numpy(value)
        else:
            self.vars[key] = torch.tensor(value)
        

    def parameters(self):
        """
        Return a list of all variables in the TensorModel.
        """
        return list(self.vars.values())
    
    
    def free_parameters(self):
        """
        Return a list of all "free" variables in the TensorModel.
        """
        return [v for v in self.vars.values() if v.requires_grad]
    
    
    def dx(self, i):
        """
        Return the size of a model element.

        Parameters
        ----------
        i: int
            Index of the dimension.

        Returns
        -------
        dx: float
            Size of the model element along axis or dimension i.
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

    
    # def create_image(self, eta, chi, axis=0):
    #     """
    #     Formal solution of the transfer equation (discretised as TensorModel)
    #     """
    #     # Check for numercial issues.
    #     dtau_max = chi.max() * self.dx(axis)
    #     if dtau_max > self.dtau_warning_threshold:
    #         print('WARNING:')
    #         print(f'  dtau_max > {self.dtau_warning_threshold}, which might lead to numerical difficulties (dtau_max = {dtau_max})!')
    #     if chi.min() < 0.0:
    #         print('WARNING:')
    #         print(f'  Negative opacities encountered, which might lead to numerical difficulties!')
    #     # Compute the optical depth and the emerging intensity.
    #     tau = self.integrate    (chi,                 axis=axis)
    #     img = self.integrate_out(eta*torch.exp(-tau), axis=axis)
    #     return img
    

    def diff(self, arr, axis=-1):
        """
        Derivative along an axis.

        Parameters
        ----------
        arr: torch.Tensor
            The array for which to compute the derivative.
        axis: int (optional)
            The axis along which to compute the derivative.

        Returns
        -------
        diff: torch.Tensor
            The derivative along the specified axis.
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

        Parameters
        ----------
        arr: torch.Tensor
            The array for which to compute the derivative.

        Returns
        -------
        diff_x: torch.Tensor
            The derivative along the x axis.
        """
        return self.diff(arr, axis=0)
    

    def diff_y(self, arr):
        """
        Derivtive along y axis.

        Parameters
        ----------
        arr: torch.Tensor
            The array for which to compute the derivative.

        Returns
        -------
        diff_y: torch.Tensor
            The derivative along the y axis.
        """
        return self.diff(arr, axis=1)

    
    def diff_z(self, arr):
        """
        Derivtive along z axis.

        Parameters
        ----------
        arr: torch.Tensor
            The array for which to compute the derivative.

        Returns
        -------
        diff_z: torch.Tensor
            The derivative along the z axis.
        """
        return self.diff(arr, axis=2)

    
    def keys(self):
        """
        Return the variable keys.
        """
        return self.vars.keys()


    def origin_as_index_array(self, origin):
        """
        Convert origin to a numpy array.

        Parameters
        ----------
        origin: array_like or str
            Indices of the origin of the coordinate system (can be float).
            The dimension of the origin should match the dimension of the model.
            If 'centre', the origin is set to the centre of the model.

        Returns
        -------
        origin: numpy array
            indices of the origin of the coordinate system.
        """
        if isinstance(origin, str) and (origin == 'centre'):
            return 0.5 * (np.array(self.shape) - 1)
        else:
            return np.array(origin)
    
    
    def get_coords(self, origin='centre'):
        """
        Getter for the coordinates of each tensor location.
        
        Parameters
        ----------
        origin: array_like
            Indices of the origin of the coordinate system (can be float).
            The dimension of the origin should match the dimension of the model.
            If 'centre', the origin is set to the centre of the model.

        Returns
        -------
        coords: numpy array
            coordinates of each location in the tensor model.
        """
        # Cast origin into a numpy array
        self.origin = self.origin_as_index_array(origin)
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

    
    def get_radius(self, origin='centre'):
        """
        Getter for the radial cooridnate of each location.

        Parameters
        ----------
        origin: array_like
            indices of the origin of the coordinate system (can be float).
            The dimension of the origin should match the dimension of the model.
            If 'centre', the origin is set to the centre of the model.
        
        Returns
        -------
        radius: numpy array
            radial coordinate of each location in the tensor model. 
        """
        # Check if coords are already set
        if (self.coords is None) or not np.all(self.origin_as_index_array(origin) == self.origin):
            self.coords = self.get_coords(origin)
        # In 1D there only is a radial coord
        if self.dimension == 1:
            return np.abs(self.coords)
        else:
            return np.linalg.norm(self.coords, axis=0)
        

    def get_radial_direction(self, origin='centre'):
        """
        Getter for the radial direction of the model.

        Parameters
        ----------
        origin: array_like (optional)
            indices of the origin of the coordinate system (can be float).
            The dimension of the origin should match the dimension of the model.

        Returns
        -------
        direction: numpy array
            radial directions of each element in the model.
        """    
        coords = self.get_coords(origin)
        radius = self.get_radius(origin)

        direction = (1.0 / radius) * coords
        
        return direction
        
        
    def apply(self, func, exclude=[], include=[]):
        """
        Apply the given functional to all model variables.

        Parameters
        ----------
        func: function
            The function to be applied to the model variables.
        exclude: list (optional)
            List of keys to exclude from the application of the function.
        include: list (optional)
            List of variables to include in the application of the function.
        
        Returns
        -------
        res: torch.Tensor
            The result of the application of the function to the model variables.
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

        Parameters
        ----------
        func: function
            The function to be applied to the model fields.
        exclude: list (optional)
            List of keys to exclude from the application of the function.
        include: list (optional)
            List of variables to include in the application of the function.

        Returns
        -------
        res: torch.Tensor
            The result of the application of the function to the model fields.
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
        "Frees" the variables that should be adjusted in optimistation.
        (and hence require a gradient.)

        Parameters
        ----------
        keys: str or list
            The key(s) of the variable(s) that should be freed.
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
        "Fixes" the variables that should not be adjusted in optimistation.
        (and hence do not require a gradient.)

        Parameters
        ----------
        keys: str or list
            The key(s) of the variable(s) that should be fixed.
        """
        # Make sure that keys are a list
        if not isinstance(keys, list):
            keys = [keys]
        # Free variables 
        for key in keys:
            self.vars[key].requires_grad_(False)
    
    def fix_all(self):
        """
        Makes all model variables fixed.
        """
        self.fix(list(self.keys()))
            
            
    def is_field(self, key):
        """
        Check if the key corresponds to a field (i.e. a variable defined at each model element).

        Parameters
        ----------
        key: str
            The key of the variable.

        Returns
        -------
        is_field: bool
            True if the key corresponds to a field, False otherwise.
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
        """
        Print info about the difference between two models.

        Parameters
        ----------
        model_1: TensorModel
            The first model.
        model_2: TensorModel
            The second model.
        """
        print("Variable:                 Min:          Mean:         Max:")
        for key in model_1.keys():
            diff = torch.abs(model_1[key] - model_2[key])
        print(f"  {key:<25} {diff.min():.3e}     {diff.mean():.3e}     {diff.max():.3e}")


    def save(self, fname):
        """
        Save the TensorModel to disk as an HDF5 file.

        Parameters
        ----------
        fname: str
            The name of the file to which the TensorModel is saved.
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

        Parameters
        ----------
        fname: str
            The name of the file from which the TensorModel is loaded.

        Returns
        -------
        model: TensorModel
            The loaded TensorModel
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

        Parameters
        ----------
        model: TensorModel
            The model from which to interpolate the variables.
        detach: bool (optional)
            If True, the interpolated tensors in this model will be detached
            from the PyTorch compute graph.
        """
        for key in model.keys():
            if model.is_field(key):
                self[key] = interpolate(model[key], self.shape).clone()
            else:
                self[key] = model[key].clone()
            # If required, detach tensor from the compute graph
            if detach:
                self.vars[key].detach_()
        


# class SphericallySymmetric():
#     """
#     Spherically Symmetric model functionality.
#     """
#     def __init__(self, model_1D):
#         """
#         Constructor for a spherically symmetric model from a 1D TensorModel.
#         """
#         # Check if the input model is in fact 1D
#         if model_1D.dimension != 1:
#             raise ValueError(f"Input model is not 1D. model_1D.dimension = {model_1D.dimension}.")

#         # Define the underlying 1D and 2D models
#         self.model_1D = model_1D
#         self.model_2D = TensorModel(
#             sizes=(2*self.model_1D.sizes[0], self.model_1D.sizes[0]),
#             shape=(2*self.model_1D.shape[0], self.model_1D.shape[0])
#         )
        
#         # Set origin for the 2D model halfway along the x axis
#         self.origin_2D = np.array([0.5*(self.model_2D.shape[0]-1), 0.0])

#         # Get linear interpolation utilities
#         self.i_min, self.i_max, self.l = self.linint_utils(self.model_2D, self.origin_2D)
    
#         # Get the circular weights
#         self.c_weights = self.get_circular_weights()


#     def linint_utils(self, model, origin):
#         """
#         Getter for the linear interpolation utilities.
#         """
#         # Compute the radii in the 2D model
#         rs = torch.from_numpy(model.get_radius(origin=origin))

#         # Compute the 2D radii in the 1D model
#         rs = (self.model_1D.shape[0] / self.model_1D.sizes[0]) * rs

#         # Compute the corresponding indices of the 2D radii in the 1D model
#         i_min = torch.floor(rs).type(torch.int64)
#         i_max = i_min + 1

#         # Bound the indices from below
#         i_min[i_min < 0] = 0
#         i_max[i_max < 0] = 0

#         # Bound the indices form above
#         i_min[i_min >= self.model_1D.shape[0]] = self.model_1D.shape[0]-1
#         i_max[i_max >= self.model_1D.shape[0]] = self.model_1D.shape[0]-1

#         # Compute the scaling factor
#         l = rs - i_min.type(torch.float64)
        
#         return i_min, i_max, l
        
         
#     def map_variables(self, keys):
#         """
#         Map a 1D Tensor Model variable to a 2D TensorModel variable assuming spherical symmetry.
#         """    
#         # Precompute the linear map
#         a = self.l
#         b = 1.0 - self.l
#         # Map variables
#         for key in keys:
#             if self.model_1D.is_field(key):
#                 self.model_2D[key] = a*self.model_1D[key][self.i_max] + b*self.model_1D[key][self.i_min]
#             else:
#                 self.model_2D[key] = self.model_1D[key]
 

#     def map_1D_to_2D(self):
#         """
#         Map a 1D TensorModel to a 2D TensorModel assuming spherical symmetry.
#         """    
#         self.map_variables(self.model_1D.keys())

        
#     def create_3D_model(self, sizes, shape, origin, detach=True):
#         """
#         Create a 3D TensorModel for the current spherically symmetric model, and given an origin.
#         If detach=True, as is by default, all model tensors will be detached form the the spherical model.
#         """
#         # Check sizes
#         if len(sizes) != 3:
#             raise ValueError("Provided sizes do not correspond to a 3D model.")
#         # Check shape
#         if len(shape) != 3:
#             raise ValueError("Provided shape does not correspond to a 3D model.")
#         # Create a 3D tensor model
#         model_3D = TensorModel(sizes=sizes, shape=shape)
#         # Compute the linear integration utilities
#         i_min, i_max, l = self.linint_utils(model_3D, origin)
#         # Precompute the linear map
#         a = l
#         b = 1.0 - l
#         # Map variables
#         for key in self.model_1D.keys():
#             if self.model_1D.is_field(key):
#                 model_3D[key] = a*self.model_1D[key][i_max] + b*self.model_1D[key][i_min]
#             else:
#                 model_3D[key] = self.model_1D[key].clone()
#             # If required, detach tensor from the compute graph
#             if detach:
#                 model_3D[key].detach_()
#         # Return the 3D TensorModel
#         return model_3D

    
#     def get_circular_weights(self):
#         """
#         Getter for the circular weights.
#         """
#         coords = self.model_2D.get_coords(origin=self.origin_2D)
#         r = coords[1,0]
#         r[0] = 0.5 * r[1]
#         return torch.from_numpy(2.0*np.pi*self.model_1D.dx(0)*r)
    
    
#     def integrate_intensity(self, img):
#         """
#         Computes the integrated intensity.
#         """
#         return torch.einsum("if,i -> f", img, self.c_weights)
    

#     def plot(self, keys=None, exclude=[]):
#         """
#         Plot the model parameters.
#         """
#         # Pick which variables to plot
#         if keys is None:
#             keys = self.model_1D.keys()
#         # Get the radial coordinate of the model
#         r = self.model_1D.get_radius(origin=[0])
#         # Plot each 1D model variable
#         for key in keys:
#             if key not in exclude:
#                 if self.model_1D.is_field(key):
#                     plt.figure(dpi=130)
#                     plt.plot(r, self.model_1D[key].data)
#                     plt.xlabel('r')
#                     plt.ylabel(key)
#                 else:
#                     print(f"{key:<21}{self.model_1D[key].data}")
        

class SphericalModel:
    """
    Spherically symmetric model.
    Convenience class to simplify creating spherically symmetic models.
    """
    def __init__(self, rs, model_1D, r_star=0.0):
        """
        Constructor for a spherically symmetric model.

        Parameters
        ----------
        rs: array_like
            The radii of the spherical model.
        model_1D: TensorModel
            The 1D model for which to construct the spherical model.
        r_star: float (optional)
            The radius of the star. (Inner boundary of the spherical model.)
        """
        self.rs       = rs            # Radii of the spherical model
        self.Nb       = len(rs) - 1   # Number of impact parameters
        self.r_star   = r_star        # Radius of the star
        self.model_1D = model_1D

        # Setup rays
        self.image_ray_tracer()


    def diff_r(self, arr, rs):
        """
        Derivative along the radial direction.

        Parameters
        ----------
        arr: torch.Tensor
            The array for which to compute the derivative.
        rs: torch.Tensor
            The radii of the spherical model.

        Returns
        -------
        diff_r: torch.Tensor
            The derivative along the radial direction.
        """
        # Prepend the one but first and append the one but last.
        # (Such that the first and last difference are the same.)
        ap0 = arr.index_select(dim=0, index=torch.tensor([0]))
        ap1 = arr.index_select(dim=0, index=torch.tensor([1]))
        am1 = arr.index_select(dim=0, index=torch.tensor([arr.size(0)-1]))
        am2 = arr.index_select(dim=0, index=torch.tensor([arr.size(0)-2]))

        pre = 2.0 * ap0 - ap1
        app = 2.0 * am1 - am2

        dr = rs.diff(prepend=torch.zeros(1))

        # Return the second-order difference.
        return (0.5 / dr) * ( arr.diff(prepend=pre) + arr.diff(append=app) )


    def get_velocity(self, model_1D):
        """
        Getter for the velocity field.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_velocity.')


    def get_temperature(self, model_1D):
        """
        Getter for the temperature distribution.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_temperature.')


    def get_abundance(self, model_1D):
        """
        Getter for the abundance distribution.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_abundance.')


    def get_turbulence(self, model_1D):
        """
        Getter for the turbulence distribution.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_turbulence.')


    def get_boundary_condition(self, model_1D, frequency, b):
        """
        Getter for the boundary condition at the impact parameter b.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_boundary_condition.')


    def image_ray_tracer(self):
        """
        Ray tracer for the spherical model.
        """
        # Initialise lists that will contain results
        self.dZss = []   # Distance increments along th ray
        self.idss = []   # Grid point indices along the ray
        self.diss = []   # Directional cosinses along the ray

        # For each impact parameter
        for i in range(self.Nb):

            Zs  = np.sqrt((self.rs[i:] - self.rs[i]) * (self.rs[i:] + self.rs[i]))
            dZs = np.diff(Zs)
            ids = np.arange(i+1, self.Nb+1)
            dis = Zs[1:] / self.rs[i+1:]

            # Check for NaNs (may occur by cancellation errors in the definition of Zs)
            if np.isnan(dZs).any():
                raise Warning('NaNs in dZs!')
            if np.isnan(dis).any():
                raise Warning('NaNs in dis!')

            if self.rs[i] >= self.r_star:
                # Append the data to the lists for this impact parameter
                self.dZss.append(torch.from_numpy(np.concatenate(( dZs[::-1],        dZs))))
                self.idss.append(torch.from_numpy(np.concatenate(( ids[::-1], [i],   ids))))
                self.diss.append(torch.from_numpy(np.concatenate((-dis[::-1], [0.0], dis))))
            else:
                # Find where the (reverse) ray is outside the star
                # Note: we need the reverse ray since our observer is at -R
                mask = (self.rs[i+1:][::-1] > self.r_star)
                # Append the data to the lists for this impact parameter
                self.dZss.append(torch.from_numpy(dZs[::-1][mask][:-1]))
                self.idss.append(torch.from_numpy(ids[::-1][mask]))
                self.diss.append(torch.from_numpy(dis[::-1][mask]))


    def image(self, lines, frequencies, r_max=np.inf, step=5):
        """
        Create synthetic image of the given spherically symmetric model.

        Parameters
        ----------
        lines: list
            List of Line objects for which to create the synthetic image.
        frequencies: list
            List of frequencies at which to compute the synthetic image.
        r_max: float (optional)
            Maximum radius at which to compute the synthetic image.
        step: int (optional)
            Step size for the impact parameters at which to trace a ray.

        Returns
        -------
        Iss: torch.Tensor
            The synthetic image of the model.
        """
        # Extract the model parameters
        velocity    = self.get_velocity   (self.model_1D)
        abundance   = self.get_abundance  (self.model_1D)
        temperature = self.get_temperature(self.model_1D)
        turbulence  = self.get_turbulence (self.model_1D)

        # Tensor for the intensities in each line
        Iss = torch.zeros((len(lines), len(frequencies[0])), dtype=torch.float64)

        # For each line
        for l, (line, freq) in enumerate(zip(lines, frequencies)):

            # Check that the number of frequencies is the same for all lines
            assert len(freq) == len(Iss[l])

            b_prev = 0.0

            # For each impact parameter
            for i in range(0, self.Nb, step):
                
                if self.rs[i] < r_max:

                    # Impact parameter
                    b = self.rs[i]

                    # Get boundary condition at this impact parameter
                    img_bdy = self.get_boundary_condition(self.model_1D, frequency=freq, b=b)

                    # Get intensity at this impact parameter
                    I_loc = line.LTE_image_along_last_axis(
                        abundance    = abundance  [self.idss[i]],
                        temperature  = temperature[self.idss[i]],
                        v_turbulence = turbulence [self.idss[i]],
                        velocity_los = velocity   [self.idss[i]] * self.diss[i],
                        frequencies  = freq,
                        dx           = self.dZss[i],
                        img_bdy      = img_bdy
                    )

                    # Surface area of the annulus at each impact parameter
                    dss = np.pi * (b + b_prev) * (b - b_prev)

                    # Integrate this piece of the annulus
                    Iss[l] += dss * I_loc

                    b_prev = b
                    
        return Iss


    def plot(self, keys=None, exclude=[]):
        """
        Plot the model parameters.
        """
        # Pick which variables to plot
        if keys is None:
            keys = self.model_1D.keys()
        # Get the radial coordinate of the model
        r = self.model_1D.get_radius(origin=[0])
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
        


class GeneralModel:
    """
    General model class.
    """
    def __init__(self, model):
        """
        Constructor for a general model.

        Parameters
        ----------
        model: TensorModel
            The model for which to create the general model.
        """
        self.model = model


    def get_velocity(self, model):
        """
        Getter for the velocity field.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_velocity.')


    def get_temperature(self, model):
        """
        Getter for the temperature distribution.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_temperature.')


    def get_abundance(self, model, l):
        """
        Getter for the abundance distribution (for the line produce species of line l).
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_abundance.')


    def get_turbulence(self, model):
        """
        Getter for the turbulence distribution.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_turbulence.')


    def get_boundary_condition(self, model, frequency):
        """
        Getter for the boundary condition.
        Has to be implemented by the user!
        """
        raise NotImplementedError('First implement and define get_boundary_condition.')


    # Forward model
    def image(self, lines, frequencies):
        """
        Create synthetic image of the given general model.

        Parameters
        ----------
        lines: list
            List of Line objects for which to create the synthetic image.
        frequencies: list

        Returns
        -------
        Iss: torch.Tensor
            The synthetic image of the model.
        """
        # Tensor for the intensities in each line
        imgs = torch.zeros((len(lines), self.model.shape[0], self.model.shape[1], len(frequencies[0])), dtype=torch.float64)

        # For each line
        for l, (line, freq) in enumerate(zip(lines, frequencies)):

            # Check that the number of frequencies is the same for all lines
            assert len(freq) == imgs[l].shape[-1]

            imgs[l] = line.LTE_image_along_last_axis(
                abundance    = self.get_abundance  (self.model, l),
                temperature  = self.get_temperature(self.model),
                v_turbulence = self.get_turbulence (self.model),
                velocity_los = self.get_velocity   (self.model)[2],
                frequencies  = freq,
                dx           = self.model.dx(0),
                img_bdy      = self.get_boundary_condition(self.model, freq)
            )

        return imgs