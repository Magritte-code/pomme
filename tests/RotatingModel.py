import healpy as hp
import torch
import math
import gc
import numpy as np
import matplotlib.pyplot as plt
from astropy        import units, constants
import astropy.constants as constants
from p3droslo.plot  import plot_cube_2D, plot_spectrum
from p3droslo.model import TensorModel
from p3droslo.lines import Line
from p3droslo.haar  import Haar
import copy
#check whether the version is correct
#the latest version should be 0.0.15, at time of writing
import p3droslo
print(p3droslo.__version__)
from astroquery.lamda import Lamda
CC  = constants.c  .si.value   # Speed of light       [m/s]
HH  = constants.h  .si.value   # Planck's constant    [J s]
KB  = constants.k_B.si.value   # Boltzmann's constant [J/K]
AMU = constants.u  .si.value   # Atomic mass unit     [kg]

class RotationModel:
    def __init__(self, model,pops ,theta, phi,padding=False):
        self.model= copy.deepcopy(model) 
        self.theta=theta
        self.phi=phi
        self.padding=padding
        
        if self.padding is True:
            self.model=self.pad_model(model)
        self.rotation_model = copy.deepcopy(self.model)  
        R,center=self.rotation_matrix(theta,phi,model)
        self.R=R
        self.center=center
        for key in model.keys():
            if key.startswith('velocity'):
                pass 
            else:
                self.rotation_model[key]=(self.rotate_scalar(self.model[key],center,R)[0])
                self.mask_region=(self.rotate_scalar(self.model[key],center,R)[1])
        if 'velocity_x:v_max' in self.rotation_model.keys() and 'velocity_y:v_max' in self.rotation_model.keys() and 'velocity_z:v_max' in self.rotation_model.keys():
            vx=self.rotate_scalar(model['velocity_x:v_max'],center,R)[0]
            vy=self.rotate_scalar(model['velocity_y:v_max'],center,R)[0]
            vz=self.rotate_scalar(model['velocity_z:v_max'],center,R)[0]
            self.rotation_model['velocity_x:v_max'], self.rotation_model['velocity_y:v_max'],self.rotation_model['velocity_z:v_max']=self.rotate_vector(vx, vy, vz, R)
        if 'velocity_los' in self.rotation_model.keys():
            self.rotation_model['velocity_los']=self.rotation_model['velocity_z:v_max']
        rotated_pops_list = []
        for i in range(pops.shape[0]):
            rotated_pop = self.rotate_scalar(pops[i,...], center, R)[0]
            rotated_pops_list.append(rotated_pop)
        self.pops = torch.stack(rotated_pops_list, dim=0)
    def __getitem__(self, key):
        # This method allows subscript notation.
        return self.rotation_model[key]
    def pad_model(self,model):
        dz=model.dx(2)
        diagonal_length = int(torch.ceil(torch.norm(torch.tensor(model.shape, dtype=torch.float)).item()))
        new_sizes=diagonal_length*dz
        padded_model= TensorModel(shape=(diagonal_length,diagonal_length,diagonal_length), sizes=3*(new_sizes,), dtau_warning_threshold=0.1)
        pad_dim1 = int(torch.ceil((diagonal_length - model.shape[0])/2))
        pad_dim2 = int(torch.ceil((diagonal_length - model.shape[1])/2))
        pad_dim3 = int(torch.ceil((diagonal_length - model.shape[2])/2))
        pads = (pad_dim3, pad_dim3, pad_dim2, pad_dim2, pad_dim1, pad_dim1)
        for key in model.keys():
            padded_model[key] = torch.nn.functional.pad(model[key], pads, 'constant', 0)
        return padded_model

    def rotation_matrix(self, theta, phi,model):
        #  z-axis phi
        R_z = torch.tensor([
            [math.cos(phi), -math.sin(phi), 0],
            [math.sin(phi),  math.cos(phi), 0],
            [0,           0,           1]
        ])
        # y-axis theta
        R_y = torch.tensor([
            [math.cos(theta), 0, math.sin(theta)],
            [0,             1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
        R =torch.matmul(R_y, R_z)
        dim_x,dim_y,dim_z=model.shape
        center = torch.zeros(3)
        center[0] = dim_x / 2
        center[1] = dim_y / 2
        center[2] = dim_z / 2
        return R,center
    def rotate_scalar(self,scalar,center,R):
        dim_x,dim_y,dim_z=scalar.shape
        rotated_scalar= torch.zeros_like(scalar)
        R_inverse=R.T
        x, y, z = torch.meshgrid(torch.arange(dim_x), torch.arange(dim_y),torch.arange(dim_z),indexing='xy')
        x_centered = x - center[0]
        y_centered = y - center[1]
        z_centered = z - center[2]
        x_old, y_old, z_old = torch.matmul(R_inverse, torch.stack([x_centered.flatten(), 
                                                                   y_centered.flatten(), 
                                                                   z_centered.flatten()]))
        x_old += center[0]
        y_old += center[1]
        z_old += center[2]
        x_old = torch.round(x_old).to(dtype=torch.int64)
        y_old = torch.round(y_old).to(dtype=torch.int64)
        z_old = torch.round(z_old).to(dtype=torch.int64)

        mask = (x_old >= 0) & (x_old < dim_x) & (y_old >= 0) & (y_old < dim_y) & (z_old >= 0) & (z_old < dim_z)
        rotated_scalar[x.ravel()[mask], 
                       y.ravel()[mask], 
                       z.ravel()[mask]] = scalar[x_old[mask], y_old[mask], z_old[mask]]
        mask_region = torch.zeros_like(scalar, dtype=bool)
        mask_region[x.ravel()[mask], y.ravel()[mask], z.ravel()[mask]] = True
        if torch.isnan(rotated_scalar).any():
            print("NaN values detected in rotate_scalar")
        return rotated_scalar,mask_region
    
    def rotate_vector(self,vx,vy,vz,R):
        #R = R.to(dtype=torch.float64)
        v = torch.stack((vx, vy, vz), axis=0)
        rotated_v = torch.einsum('ij,jklm->iklm', R, v)
        r_vx = rotated_v[0]
        r_vy = rotated_v[1]
        r_vz = rotated_v[2]
        return r_vx,r_vy,r_vz
    
