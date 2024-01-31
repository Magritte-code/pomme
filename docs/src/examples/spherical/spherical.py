import numpy as np
import torch

from astropy        import units, constants
from p3droslo.lines import Line
from p3droslo.loss  import Loss, diff_loss
from p3droslo.model import TensorModel, SphericalModel
from p3droslo.utils import print_var, planck


n_elements = 128

r_in   = (1.0e-1 * units.au).si.value
r_out  = (1.0e+4 * units.au).si.value

v_in  = (1.0e+0 * units.km / units.s).si.value
v_inf = (2.0e+1 * units.km / units.s).si.value
beta  = 0.5

T_in    = (5.0e+3 * units.K).si.value
epsilon = 0.3

Mdot   = (1.0e-3 * units.M_sun / units.yr).si.value
v_turb = (1.5e+0 * units.km    / units.s ).si.value
T_star = (1.0e+4 * units.K               ).si.value
R_star = (1.0e+0 * units.au              ).si.value

model = TensorModel(sizes=r_out, shape=n_elements)

rs = np.logspace(np.log10(r_in), np.log10(r_out), n_elements, dtype=np.float64)
# rs = np.linspace(         r_in ,          r_out , n_elements, dtype=np.float64)

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

# Line data
lines = [Line('CO', i) for i in range(10)]

# Frequency data
vdiff = 300   # velocity increment size [m/s]
nfreq = 100   # number of frequencies

velocities  = nfreq * vdiff * torch.linspace(-1, +1, nfreq, dtype=torch.float64)
frequencies = [(1.0 + velocities / constants.c.si.value) * line.frequency for line in lines]


def get_velocity(model):

    r      = torch.exp(model['log_r'])
    v_in   = torch.exp(model['log_v_in'])
    v_inf  = torch.exp(model['log_v_inf'])
    beta   = torch.exp(model['log_beta'])
    R_star = torch.exp(model['log_R_star'])

    v = torch.empty_like(r)
    v[r <= R_star] = 0.0
    v[r >  R_star] = v_in + (v_inf - v_in) * (1.0 - R_star / r[r > R_star])**beta
    # Return
    return v


def get_temperature(model):

    r       = torch.exp(model['log_r'])
    T_in    = torch.exp(model['log_T_in'])
    epsilon = torch.exp(model['log_epsilon'])
    R_star  = torch.exp(model['log_R_star'])
    
    T = torch.empty_like(r)    
    T[r <= R_star] = T_in
    T[r >  R_star] = T_in * (R_star / r[r > R_star])**epsilon
    # Return
    return T


def get_abundance(model):
    return torch.exp(model['log_CO'])


def get_turbulence(model):
    return torch.exp(model['log_turbulence'])


def get_boundary_condition(model, frequency, b):

        T_star = torch.exp(model['log_T_star'])
        R_star = torch.exp(model['log_R_star'])

        T_CMB = 2.72548

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