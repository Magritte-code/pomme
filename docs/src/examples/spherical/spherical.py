import numpy as np
import torch

from astropy     import units, constants
from pomme.lines import Line
from pomme.loss  import Loss, diff_loss
from pomme.model import TensorModel, SphericalModel
from pomme.utils import planck, T_CMB


n_elements = 128

r_in   = (1.0e-1 * units.au).si.value
r_out  = (1.0e+4 * units.au).si.value

v_in  = (1.0e+0 * units.km / units.s).si.value
v_inf = (2.0e+1 * units.km / units.s).si.value
beta  = 0.5

T_in    = (2.5e+3 * units.K).si.value
epsilon = 0.6

Mdot   = (1.0e-6 * units.M_sun / units.yr).si.value
v_turb = (1.5e+0 * units.km    / units.s ).si.value
T_star = (2.5e+3 * units.K               ).si.value
R_star = (1.0e+0 * units.au              ).si.value

model = TensorModel(sizes=r_out, shape=n_elements)

rs = np.logspace(np.log10(r_in), np.log10(r_out), n_elements, dtype=np.float64)

v = np.empty_like(rs)
v[rs <= R_star] = 0.0
v[rs >  R_star] = v_in + (v_inf - v_in) * (1.0 - R_star / rs[rs > R_star])**beta

rho  = Mdot / (4.0 * np.pi * rs**2 * v)
n_CO = (3.0e-4 * constants.N_A.si.value / 2.02e-3) * rho
n_CO[rs<=R_star] = n_CO[n_CO<np.inf].max()


# Define and initialise the model variables
model['log_r'         ] = np.log(rs)
model['log_CO'        ] = np.log(n_CO)
model['log_turbulence'] = np.log(v_turb) * np.ones(n_elements)
model['log_v_in'      ] = np.log(v_in)
model['log_v_inf'     ] = np.log(v_inf)
model['log_beta'      ] = np.log(beta)
model['log_T_in'      ] = np.log(T_in)
model['log_epsilon'   ] = np.log(epsilon)
model['log_T_star'    ] = np.log(T_star)
model['log_R_star'    ] = np.log(R_star)

model.fix_all()
model.save('model_truth.h5')

# Line data
lines = [Line('CO', i) for i in [0, 1, 3, 5]]

# Frequency data
vdiff = 300   # velocity increment size [m/s]
nfreq = 100   # number of frequencies

velocities  = nfreq * vdiff * torch.linspace(-1, +1, nfreq, dtype=torch.float64)
frequencies = [(1.0 + velocities / constants.c.si.value) * line.frequency for line in lines]


def get_velocity(model):
    """
    Get the velocity from the TensorModel.
    """
    # Extract parameters
    r      = torch.exp(model['log_r'])
    v_in   = torch.exp(model['log_v_in'])
    v_inf  = torch.exp(model['log_v_inf'])
    beta   = torch.exp(model['log_beta'])
    R_star = torch.exp(model['log_R_star'])
    # Compute velocity
    v = torch.empty_like(r)
    v[r <= R_star] = 0.0
    v[r >  R_star] = v_in + (v_inf - v_in) * (1.0 - R_star / r[r > R_star])**beta
    # Return
    return v


def get_temperature(model):
    """
    Get the temperature from the TensorModel.
    """
    # Extract parameters
    r       = torch.exp(model['log_r'])
    T_in    = torch.exp(model['log_T_in'])
    epsilon = torch.exp(model['log_epsilon'])
    R_star  = torch.exp(model['log_R_star'])
    # Compute temperature
    T = torch.empty_like(r)    
    T[r <= R_star] = T_in
    T[r >  R_star] = T_in * (R_star / r[r > R_star])**epsilon
    # Return
    return T


def get_abundance(model):
    """
    Get the abundance from the TensorModel.
    """
    return torch.exp(model['log_CO'])


def get_turbulence(model):
    """
    Get the turbulence from the TensorModel.
    """
    return torch.exp(model['log_turbulence'])


def get_boundary_condition(model, frequency, b):
    """
    Get the boundary condition from the TensorModel.
    """
    # Extract parameters
    T_star = torch.exp(model['log_T_star'])
    R_star = torch.exp(model['log_R_star'])
    # Compute boundary condition
    if b > R_star:
        return planck(temperature=T_CMB, frequency=frequency)
    else:
        return planck(temperature=T_star, frequency=frequency)


smodel = SphericalModel(rs, model, r_star=R_star)
smodel.get_velocity           = get_velocity
smodel.get_abundance          = get_abundance
smodel.get_turbulence         = get_turbulence
smodel.get_temperature        = get_temperature
smodel.get_boundary_condition = get_boundary_condition