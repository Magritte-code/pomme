# %% [markdown]
# # Reconstruction - Phantom model
# ---

# %%
import torch
import numpy             as np
import matplotlib.pyplot as plt

from ipywidgets  import interact
from torch.optim import Adam
from tqdm        import tqdm
from astropy     import constants, units
from pomme.model import TensorModel
from pomme.loss  import Loss, diff_loss, SphericalLoss
from pomme.lines import Line
from pomme.plot  import plot_cube_2D
from pomme.utils import planck, T_CMB

from phantom import lines,fracs, velos, Model

# %%
model_truth = TensorModel.load('model_truth.h5')

# %%
model = TensorModel(sizes=model_truth.sizes, shape=model_truth.shape)

# %%
rs = model.get_radius(origin='centre')

v_in  = 1.0e+2
v_inf = 1.0e+4
beta  = 1.0

T_in    = 5.0e+3
epsilon = 0.5

# Initialise
model['log_H2'          ] = np.log(5.0e+13 * (rs.min()/rs)**2)
# model['log_v_in'        ] = np.log(v_in)
# model['log_v_inf'       ] = np.log(v_inf)
# model['log_beta'        ] = np.log(beta)
model['log_velocity_r'  ] = np.log(v_in + (v_inf - v_in) * (1.0 - rs.min() / rs)**beta)
# model['log_T_in'        ] = np.log(T_in)
# model['log_epsilon'     ] = np.log(epsilon)
model['log_temperature'] = np.log(T_in * (rs.min()/rs)**epsilon)
model['log_v_turbulence'] = np.log(1.5e+2) * np.ones(model.shape, dtype=np.float64)

# model.free(['log_H2', 'log_v_in', 'log_v_inf', 'log_beta', 'log_T_in', 'log_epsilon'])
# model.free(['log_H2', 'log_velocity_r', 'log_T_in', 'log_epsilon'])
model.free(['log_H2', 'log_velocity_r', 'log_temperature'])

# losses = Loss(['avg', 'rel', 'reg', 'cnt'])
# losses = Loss(['avg', 'rel', 'reg'])
losses = Loss(['avg', 'rel'])

# %%
rs.min() / (1.0 * units.au).si.value

# %%
# def get_velocity(model):

#     v_in  = torch.exp(model['log_v_in' ])
#     v_inf = torch.exp(model['log_v_inf'])
#     beta  = torch.exp(model['log_beta' ])
    
#     r = torch.from_numpy(model.get_radius(origin='centre'))
#     d = torch.from_numpy(model.get_radial_direction(origin='centre'))
    
#     return (v_in + (v_inf - v_in) * (1.0 - r.min() / r)**beta) * d


def get_velocity(model):
    
    d = torch.from_numpy(model.get_radial_direction(origin='centre'))
    
    return torch.exp(model['log_velocity_r']) * d


# def get_temperature(model):

#     T_in    = torch.exp(model['log_T_in'])
#     epsilon = torch.exp(model['log_epsilon'])    
    
#     r = torch.from_numpy(model.get_radius(origin='centre'))
    
#     return T_in * (r.min() / r)**epsilon

def get_temperature(model):
    
    return torch.exp(model['log_temperature'])


def get_boundary_condition(model, freq):
    Ibdy  = torch.ones((model.shape[0], model.shape[1], len(freq)), dtype=torch.float64)
    Ibdy *= planck(temperature=T_CMB, frequency=freq)
    return Ibdy


pmodel = Model(model, lines, fracs, velos)
pmodel.get_velocity           = get_velocity
pmodel.get_abundance          = lambda model: torch.exp(model['log_H2'])
pmodel.get_turbulence         = lambda model: torch.exp(model['log_v_turbulence'])
pmodel.get_temperature        = get_temperature
pmodel.get_boundary_condition = get_boundary_condition

# %%
obss = torch.load('obss.pt')
imgs = pmodel.image()

# %%
obss_int = obss.sum(axis=(1,2))
imgs_int = imgs.sum(axis=(1,2))

plt.plot(imgs_int[0].data, c='tab:blue')
plt.plot(obss_int[0].data, c='tab:blue', linestyle='--')
plt.plot(imgs_int[1].data, c='tab:orange')
plt.plot(obss_int[1].data, c='tab:orange', linestyle='--')

# %%
# def steady_state_continuity_loss(pmodel):
#     """
#     Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives in Euler's equations.
#     """
#     rho           = pmodel.get_abundance(pmodel.model)
#     v_x, v_y, v_z = pmodel.get_velocity (pmodel.model)
    
#     # Continuity equation (steady state): div(ρ v) = 0
#     loss_cont = pmodel.model.diff_x(rho * v_x) + pmodel.model.diff_y(rho * v_y) + pmodel.model.diff_z(rho * v_z)
#     # Squared average over the model
#     loss_cont = torch.mean((loss_cont / rho)**2)

#     return loss_cont

# %%
from torch.optim import Adam
from tqdm        import tqdm


def fit(losses, pmodel, obss, N_epochs=10, lr=1.0e-1, w_avg=1.0, w_rel=1.0, w_reg=1.0, w_cnt=1.0):

    params = [
        pmodel.model['log_H2'],
        # pmodel.model['log_v_in'],
        # pmodel.model['log_v_inf'],
        # pmodel.model['log_beta'],
        pmodel.model['log_velocity_r'],
        # pmodel.model['log_T_in'],
        # pmodel.model['log_epsilon'],
        pmodel.model['log_temperature'],
    ]
    
    optimizer = Adam(params, lr=lr)

    obss_avg= obss.mean(axis=-1)
    obss_rel= torch.einsum("...f, ... -> ...f", obss, 1.0 / obss_avg)

    for _ in tqdm(range(N_epochs)):

        # Forward model
        imgs = pmodel.image()

        imgs_avg= imgs.mean(axis=-1)
        imgs_rel= torch.einsum("...f, ... -> ...f", imgs, 1.0 / imgs_avg)

        # Compute the reproduction loss
        losses['avg'] = w_avg * torch.nn.functional.mse_loss(torch.log(imgs_avg), torch.log(obss_avg))
        losses['rel'] = w_rel * torch.nn.functional.mse_loss(          imgs_rel,            obss_rel )
        # Compute the regularisation loss
        # losses['reg'] = w_reg * (  diff_loss(pmodel.model['log_H2'])
                                #  + diff_loss(pmodel.model['log_velocity_r'])
                                #  + diff_loss(pmodel.model['log_temperature'])
                                # )
        # Compute the hydrodynamic loss   
        # losses['cnt'] = w_cnt * steady_state_continuity_loss(pmodel)   

        # Set gradients to zero
        optimizer.zero_grad()
        # Backpropagate gradients
        losses.tot().backward()
        # Update parameters
        optimizer.step()

    return imgs, losses

# %%
pmodel.model.info()

# %%
pmodel.model.save('model_fit_it_no_con_no_diff=000.h5')

# %%
imgs, losses = fit(losses, pmodel, obss, N_epochs=3, lr=1.0e-1, w_avg=1.0, w_rel=1.0e+0, w_reg=1.0e-0, w_cnt=1.0e+0)

losses.renormalise_all()
losses.reset()

# %%
imgs, losses = fit(losses, pmodel, obss, N_epochs=100, lr=1.0e-1, w_avg=1.0, w_rel=1.0e+0, w_reg=1.0e-0, w_cnt=1.0e+0)

# %%
losses.plot()

# %%
pmodel.model.save('model_fit_it_no_con_no_diff=100.h5')

# %%
imgs, losses = fit(losses, pmodel, obss, N_epochs=100, lr=1.0e-1, w_avg=1.0e+1, w_rel=1.0e+1, w_reg=1.0e+1, w_cnt=1.0e+1)

# %%
losses.plot()

# %%
pmodel.model.save('model_fit_it_no_con_no_diff=200.h5')

# %%
imgs, losses = fit(losses, pmodel, obss, N_epochs=100, lr=1.0e-1, w_avg=1.0e+2, w_rel=1.0e+2, w_reg=1.0e+2, w_cnt=1.0e+2)

# %%
losses.plot()

# %%
pmodel.model.save('model_fit_it_no_con_no_diff=300.h5')

# %%
imgs, losses = fit(losses, pmodel, obss, N_epochs=100, lr=1.0e-1, w_avg=1.0e+3, w_rel=1.0e+3, w_reg=1.0e+3, w_cnt=1.0e+3)

# %%
losses.plot()

# %%
pmodel.model.save('model_fit_it_no_con_no_diff=400.h5')

# %%
# pmodel.model.load('model_fit_it=400.h5')
# pmodel.model.info()

# %%
imgs, losses = fit(losses, pmodel, obss, N_epochs=100, lr=1.0e-1, w_avg=1.0e+4, w_rel=1.0e+4, w_reg=1.0e+1, w_cnt=1.0e+3)

# %%
losses.plot()

# %%
pmodel.model.save('model_fit_it_no_con_no_diff=500.h5')

# %%
imgs, losses = fit(losses, pmodel, obss, N_epochs=100, lr=1.0e-1, w_avg=1.0e+5, w_rel=1.0e+5, w_reg=1.0e+2, w_cnt=1.0e+3)

# %%
losses.plot()

# %%
pmodel.model.save('model_fit_it_no_con_no_diff=600.h5')

# %%



