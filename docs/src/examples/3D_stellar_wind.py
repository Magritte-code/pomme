# %% [markdown]
# # 3D Stellar wind
# 
# ---
# In this example, we consider an smoothed-particle hydrodynamics (SPH) model of a companion-perturbed stellar wind.

# %%
import numpy             as np
import matplotlib.pyplot as plt
import os
from astropy import units, constants

# %% [markdown]
# 
# ---
# 
# ## Model setup
# First we download a snapshot of this SPH model.

# %%
setup_file = '3D_stellar_wind_data/wind.setup'
input_file = '3D_stellar_wind_data/wind.in'
dump_file  = '3D_stellar_wind_data/wind.dump'

# %%
# !wget 'https://raw.githubusercontent.com/Ensor-code/phantom-models/main/Malfait%2B2024a/v10e00/wind.setup'  --output-document $setup_file
# !wget 'https://raw.githubusercontent.com/Ensor-code/phantom-models/main/Malfait%2B2024a/v10e00/wind.in'     --output-document $input_file
# !wget 'https://raw.githubusercontent.com/Ensor-code/phantom-models/main/Malfait%2B2024a/v10e00/wind_v10e00' --output-document $dump_file

# %% [markdown]
# We use [plons](https://github.com/Ensor-code/plons) to open the data.

# %%
import plons

setupData = plons.LoadSetup(f'{os.getcwd()}/3D_stellar_wind_data', 'wind')
dumpData  = plons.LoadFullDump(f'{os.getcwd()}/{dump_file}', setupData)

position = dumpData["position"]*1e-2   # position vectors [cm   -> m]
velocity = dumpData["velocity"]*1e3    # velocity vectors [km/s -> m/s]
rho      = dumpData["rho"]             # density          [g/cm^3]
tmp      = dumpData["Tgas"]            # temperature      [K]
tmp[tmp<2.7] = 2.7                     # Cut-off temperatures below 2.7 K

# Unpack velocities
v_x, v_y, v_z = velocity.T

# Convert rho (total density) to H2 abundance
nH2 = rho * 1.0e+6 * constants.N_A.si.value / 2.02

# Define turbulence at 150 m/s
v_trb = 150.0

# %% [markdown]
# Next, we map the particle data to a regular Cartesian mesh.

# %%
from pomme.haar import Haar

# Map point data to a regular grid
haar = Haar(position, q=8)
# Zoom in on the centre region to avoid edge effects
imin = 2**(haar.q-3)
imax = 3*imin
# Map data to a regular grid
nH2_dat = haar.map_data(nH2, interpolate=True)[-1][imin:imax,imin:imax,imin:imax]
tmp_dat = haar.map_data(tmp, interpolate=True)[-1][imin:imax,imin:imax,imin:imax]
v_x_dat = haar.map_data(v_x, interpolate=True)[-1][imin:imax,imin:imax,imin:imax]
v_y_dat = haar.map_data(v_y, interpolate=True)[-1][imin:imax,imin:imax,imin:imax]
v_z_dat = haar.map_data(v_z, interpolate=True)[-1][imin:imax,imin:imax,imin:imax]

# %% [markdown]
# ---
# 
# ### TensorModel
# With all the data in place, we can start building a pomme model.
# First, we store all model parameters as a TensorModel object and store this in an HDF5 file.
# We will use this later as the ground truth to verify our reconstructions against.

# %%
from pomme.model import TensorModel

model = TensorModel(shape=nH2_dat.shape, sizes=haar.xyz_L)
model['log_H2'          ] = np.log(nH2_dat).astype(np.float64)
model['log_temperature' ] = np.log(tmp_dat).astype(np.float64)
model['velocity_x'      ] =        v_x_dat .astype(np.float64)
model['velocity_y'      ] =        v_y_dat .astype(np.float64)
model['velocity_z'      ] =        v_z_dat .astype(np.float64)
model['log_v_turbulence'] = np.log(v_trb)*np.ones(model.shape, dtype=np.float64)
model.save('3D_stellar_wind_truth.h5')

# %% [markdown]
# ---
# 
# ### GeneralModel
# First, we define the functions that can generate the model distributions from the model parameters.

# %%
import torch
from pomme.utils import planck, T_CMB

def get_velocity(model):
    """
    Get the velocity from the TensorModel.
    """
    return model['velocity_z']

def get_temperature(model):
    """
    Get the temperature from the TensorModel.
    """
    return torch.exp(model['log_temperature'])

def get_abundance(model, l):
    """
    Get the abundance from the TensorModel.
    """
    # Define the assumed molecular fractions w.r.t. H2
    X_mol = [3.0e-4, 5.0e-6]
    # Return the abundance
    return torch.exp(model['log_H2']) * X_mol[l]

def get_turbulence(model):
    """
    Get the turbulence from the TensorModel.
    """
    return torch.exp(model['log_v_turbulence'])

def get_boundary_condition(model, frequency):
    """
    Get the boundary condition from the TensorModel.
    model: TensorModel
        The TensorModel object containing the model.
    frequency: float
        Frequency at which to evaluate the boundary condition.
    """
    # Compute the incoming boundary intensity
    Ibdy  = torch.ones((model.shape[0], model.shape[1], len(frequency)), dtype=torch.float64)
    Ibdy *= planck(temperature=T_CMB, frequency=frequency)
    # Return the result
    return Ibdy

# %% [markdown]
# Using these functions, we can build a GeneralModel object that can be used to generate synthetic observations or reconstruct the required parameters. (Cfr. the SphericalModel class for spherically symmetric models.)

# %%
from pomme.model import GeneralModel

gmodel_truth = GeneralModel(model=model)
gmodel_truth.get_velocity           = get_velocity
gmodel_truth.get_abundance          = get_abundance
gmodel_truth.get_turbulence         = get_turbulence
gmodel_truth.get_temperature        = get_temperature
gmodel_truth.get_boundary_condition = get_boundary_condition

# %% [markdown]
# ---
# 
# ### Spectral lines
# We base our reconstructions on synthetic observations of two commonly observed rotational lines CO $J=4-3$ and SiO $J=3-2$. We explicitly provide the molar mass for SiO, since this is not extracted correctly from the line data file.

# %%
from pomme.lines import Line
from pomme.utils import get_molar_mass

lines = [
    Line(species_name='CO',     transition=3),
    Line(species_name='sio-h2', transition=2, molar_mass=get_molar_mass('SiO'))
]

# %% [markdown]
# ---
# 
# ### Frequencies
# Next, we define the velocity/frequency range.
# We observe the lines in 100 frequency bins, centred around the lines, with a spacing of 120 m/s.

# %%
vdiff = 120   # velocity increment size [m/s]
nfreq = 100   # number of frequencies

velocities  = nfreq * vdiff * torch.linspace(-1, +1, nfreq, dtype=torch.float64)
frequencies = [(1.0 + velocities / constants.c.si.value) * line.frequency for line in lines]

# %% [markdown]
# ---
# 
# ## Synthetic observations
# We can now generate synthetic observations, directly from the Model object.
# We will use these later to derive our reconstructions.

# %%
obss = gmodel_truth.image(lines, frequencies)

# %% [markdown]
# Plot the resulting synthetic spectral line observations.
# pomme provides some convenient wrappers for matploltib to explore the spectral data cubes.

# %%
from pomme.plot import plot_cube_2D, plot_spectrum

plot_cube_2D(np.log(obss[0]))
plot_cube_2D(np.log(obss[1]))

plot_spectrum(np.log(obss[0]))
plot_spectrum(np.log(obss[1]))

# %% [markdown]
# ---
# 
# ## Reconstruction setup
# In this example, we will try to reconstruct the CO abundance, radial velocity, and temperature distribution.
# Since we will only assume a radial component in our velocity field, we need to adapt the corresponding function.

# %%
def get_velocity_from_1D(model):
    """
    Get the velocity from the 1D data.
    """ 
    # Extract the radial unit vector field from the model
    d = torch.from_numpy(model.get_radial_direction(origin='centre'))
    # Return the (strictly radial) velocity field
    return torch.exp(model['log_velocity_r']) * d

# %% [markdown]
# Next, we define the model object for the reconstruction. Note that in the gmodel defined above, all the right parameters are already stored, so we need a new one for the reconstruction.
# We take, $n_{\text{CO}}^{\text{init}}(r) = 5.0 \times 10^{14} \, \text{m}^{-3} \, (r_{\text{in}}/r)^{2} $, as initial guess for the CO abundance distribution, and initialise both the velocity and the temperature with the correct values.

# %%
gmodel_recon = GeneralModel(gmodel_truth.model)
gmodel_recon.get_velocity           = get_velocity_from_1D
gmodel_recon.get_abundance          = get_abundance
gmodel_recon.get_turbulence         = get_turbulence
gmodel_recon.get_temperature        = get_temperature
gmodel_recon.get_boundary_condition = get_boundary_condition

# Initial guess parameters
n_H2_in = 5.0e+13
v_in    = 1.0e+2
v_inf   = 1.0e+4
beta    = 1.0
T_in    = 5.0e+3
epsilon = 0.5

rs = gmodel_recon.model.get_radius(origin='centre')

model['log_H2'          ] = np.log(n_H2_in * (rs.min()/rs)**2)
model['log_velocity_r'  ] = np.log(v_in + (v_inf - v_in) * (1.0 - rs.min()/rs)**beta)
model['log_temperature' ] = np.log(T_in * (rs.min()/rs)**epsilon)

# Fix all parameters, except for the ones we want to fit
model.fix_all()
model.free(['log_H2', 'log_velocity_r', 'log_temperature'])

# Save the initial guess
model.save('3D_stellar_wind_recon_init.h5')

# %% [markdown]
# We can explore the model parameters with the info() function.

# %%
gmodel_recon.model.info()

# %% [markdown]
# ---
# 
# ### Loss functions
# We first create Loss object that can conveniently store the different losses.
# We will use a reproduction loss (split into an averaged and relative component), a smoothness, and a continuity loss.

# %%
from pomme.loss import Loss, diff_loss

losses = Loss(['avg', 'rel', 'smt', 'cnt'])

# %% [markdown]
# ---
# 
# #### Reproduction loss
# We split the reproduction loss into an averaged and a relative component,
# \begin{equation*}
# \mathcal{L}_{\text{rep}}\big(f(\boldsymbol{m}), \boldsymbol{o} \big)
# \ = \
# \mathcal{L}_{\text{rep}}\Big( \big\langle f(\boldsymbol{m}) \big\rangle, \, \left\langle\boldsymbol{o}\right\rangle \Big)
# \ + \
# \mathcal{L}_{\text{rep}}\left( \frac{f(\boldsymbol{m})}{\big\langle f(\boldsymbol{m})\big\rangle}, \, \frac{\boldsymbol{o}}{\left\langle \boldsymbol{o}\right\rangle} \right) ,
# \end{equation*}

# %%
# Define averaging and relative function
avg = lambda arr: arr.mean(axis=-1)
rel = lambda arr: torch.einsum("...f, ... -> ...f", arr, 1.0/avg(arr))

def avg_loss(smodel, imgs):
    """
    Compute the average loss.
    """
    return torch.nn.functional.mse_loss(avg(imgs), avg(obss))

def rel_loss(smodel, imgs):
    """
    Compute the relative loss.
    """
    return torch.nn.functional.mse_loss(rel(imgs), rel(obss))

# %% [markdown]
# ---
# 
# #### Smoothness loss
# The smoothnes loss is defined as,
# \begin{equation}
# \mathcal{L}[q] \ = \ \int \text{d} \boldsymbol{x} \ \| \nabla q(\boldsymbol{x})\|^{2}
# \end{equation}

# %%
def smoothness_loss(gmodel):
    """
    Smoothness loss for CO, velocity, and temperature distributions.
    """
    # Compute and return the loss
    return (   diff_loss(gmodel.model['log_H2'         ]) \
             + diff_loss(gmodel.model['log_velocity_r' ]) \
             + diff_loss(gmodel.model['log_temperature']) )

# %% [markdown]
# ---
# 
# #### Continuity loss
# The regularisation loss that assumes a steady state and enforces the continuity equation, in 3D reads,
# \begin{equation*}
# \mathcal{L}[\rho, \boldsymbol{v}]
# \ = \
# \int \text{d}\boldsymbol{x} \ \frac{1}{\rho^{2}} \big( \nabla \cdot \left( \rho \, \boldsymbol{v} \right) \big)^{2}
# \end{equation*}

# %%
def steady_state_cont_loss(gmodel):
    """
    Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives.
    """
    # Extract the relevant variables form the model
    rho           = gmodel.get_abundance(gmodel.model, 0)   # line number does not matter here
    v_x, v_y, v_z = gmodel.get_velocity (gmodel.model)
    # Continuity equation (steady state): div(ρ v) = 0
    loss_cont = gmodel.model.diff_x(rho * v_x) + gmodel.model.diff_y(rho * v_y) + gmodel.model.diff_z(rho * v_z)
    # Squared average over the model
    loss_cont = torch.mean((loss_cont / rho)**2)
    # Return losses
    return loss_cont

# %% [markdown]
# ---
# 
# ### Fit function
# With everything in place, we can finally define the fit function.

# %%
from torch.optim import Adam
from tqdm        import tqdm

def fit(losses, gmodel, N_epochs=10, lr=1.0e-1, w_avg=1.0, w_rel=1.0, w_smt=1.0, w_cnt=1.0):
    # Define optimiser
    optimizer = Adam(model.free_parameters(), lr=lr)
    # Iterate optimiser
    for _ in tqdm(range(N_epochs)):
        # Forward model
        imgs = gmodel.image(lines, frequencies)
        # Compute the losses
        losses['avg'] = w_avg * avg_loss(gmodel, imgs)
        losses['rel'] = w_rel * rel_loss(gmodel, imgs)
        losses['smt'] = w_smt * smoothness_loss(gmodel)
        losses['cnt'] = w_cnt * steady_state_cont_loss(gmodel)
        # Set gradients to zero
        optimizer.zero_grad()
        # Backpropagate gradients
        losses.tot().backward()
        # Update parameters
        optimizer.step()
    # Return the images and losses
    return imgs, losses

# %% [markdown]
# ---
# 
# ## Experiments

# %% [markdown]
# Now we can fit the model to the synthetic data.
# First, to determine the relative weights for each loss function in the total loss, we run 3 iterations and observe the loss values. Then, we define the weight of each individual loss function by the inverse of its current value (renormalisation step), such that in the following iterations they all contribute equally.

# %%
imgs, losses = fit(losses, gmodel_recon,
    N_epochs = 3,
    lr       = 1.0e-1,
    w_avg    = 1.0e+0,
    w_rel    = 1.0e+0,
    w_smt    = 1.0e+0,
    w_cnt    = 1.0e+0,
)
losses.renormalise_all()   # Renormalise the losses
losses.reset()             # Reset the losses to renormalised values

# Save the (partially) reconstructed model
gmodel_recon.model.save('3D_stellar_wind_recon_000.h5')

# %% [markdown]
# With the renormalised weights in place, we now run 100 iterations of the reconstruction algorithm.
# Note that one can still specify relative weights for the losses, these are an additional factor to the renormalisation. Here, we take them all equal to one, such that all losses contribute equally.

# %%
imgs, losses = fit(losses, gmodel_recon,
    N_epochs = 100,
    lr       = 1.0e-1,
    w_avg    = 1.0e+0,
    w_rel    = 1.0e+0,
    w_smt    = 1.0e+0,
    w_cnt    = 1.0e+0,
)
gmodel_recon.model.save('3D_stellar_wind_recon_100.h5')
losses.plot()

# %% [markdown]
# Next, we run an additional 500 itereations of the reconstruction algortihm.

# %%
imgs, losses = fit(losses, gmodel_recon,
    N_epochs = 500,
    lr       = 1.0e-1,
    w_avg    = 1.0e+0,
    w_rel    = 1.0e+0,
    w_smt    = 1.0e+0,
    w_cnt    = 1.0e+0,
)
gmodel_recon.model.save('3D_stellar_wind_recon_600.h5')
losses.plot()


