import matplotlib.pyplot as plt
import numpy             as np
import torch

from torch.optim          import Adam
from tqdm                 import tqdm
from astropy              import units, constants

from pomme.model       import TensorModel, SphericallySymmetric
from pomme.lines       import Line
from pomme.loss        import Loss, diff_loss

from test_1D import forward, get_model, line, frequencies, velocities, r_in


def get_initial_model(from_model, nCO, tmp):

    # Define and initialise the model variables
    model_1D = TensorModel.load(from_model)
    model_1D['log_CO'         ] = np.log(nCO) * np.ones(model_1D.shape)
    model_1D['log_temperature'] = np.log(tmp) * np.ones(model_1D.shape)

    # Explicitly set all model variables fixed except log_CO
    model_1D.fix_all()
    model_1D.free('log_CO')
    model_1D.free('log_temperature')

    # Create a spherically symmetric model form the 1D TensorModel
    spherical = SphericallySymmetric(model_1D)

    return spherical


def steady_state_cont_loss(spherical, r):
    """
    Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives.
    """
    log_rho = spherical.model_1D['log_CO']
    log_v_r = spherical.model_1D['log_velocity']

    rho = torch.exp(log_rho)         
    v_r = torch.exp(log_v_r)
    
    # Continuity equation (steady state): div(ρ v) = 0
    loss_cont = spherical.model_1D.diff_x(r**2 * rho * v_r)

    # Compute the mean squared losses
    loss = torch.mean((loss_cont/((r**2)*rho))**2)

    # Return losses
    return loss


def steady_state_heat_loss(spherical, r):
    """
    Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives.
    """
    log_tmp = spherical.model_1D['log_temperature']
    log_rad = torch.log(r)

    log_tmp = log_tmp - log_tmp[0]
    log_rad = log_rad - log_rad[0]

    loss_heat = spherical.model_1D.diff_x(log_tmp[1:] / log_rad[1:]) 
    # Heat equation (steady state): div(ρ v) = 0
    # loss_heat = spherical.model_1D.diff_x(r * spherical.model_1D.diff_x(log_tmp))

    # Compute the mean squared losses
    loss = torch.mean(loss_heat**2)

    # Return losses
    return loss


def fit(loss, spherical, obs, N_epochs=10, lr=1.0e-1, w_rep=1.0, w_reg=1.0, w_tmp=1.0):

    r = spherical.model_1D.get_coords(origin=np.array([-2]))
    r[r<r_in] = r_in

    optimizer = Adam(spherical.model_1D.free_parameters(), lr=lr)

    for _ in tqdm(range(N_epochs)):
        
        # Run forward model
        img = forward(spherical)
 
        # Compute the reproduction loss
        loss['rep'] = w_rep * torch.nn.functional.mse_loss(img, obs)
        # Compute the regularisation loss
        loss['reg'] = w_reg * spherical.model_1D.apply(diff_loss)
        # Compute the hydrodynamic loss
        # loss['tmp'] = w_tmp * steady_state_heat_loss(spherical, torch.from_numpy(r))    

        # Set gradients to zero
        optimizer.zero_grad()
        # Backpropagate gradients
        loss.tot().backward()
        # Update parameters
        optimizer.step()

    return img


def reconstruct(spherical, obs):

    loss = Loss(['rep', 'reg'])
    # loss = Loss(['rep', 'reg', 'tmp'])

    img = fit(loss, spherical, obs, N_epochs=3  , lr=1.0e-1, w_rep=1.0e+0, w_reg=1.0e-0, w_tmp=1.0e+0)
    loss.renormalise_all()
    loss.reset()
    img = fit(loss, spherical, obs, N_epochs=250, lr=5.0e-1, w_rep=1.0e+0, w_reg=1.0e-2, w_tmp=1.0e+0)
    img = fit(loss, spherical, obs, N_epochs=250, lr=1.0e-1, w_rep=1.0e+1, w_reg=1.0e-1, w_tmp=1.0e+1)
    img = fit(loss, spherical, obs, N_epochs=250, lr=5.0e-2, w_rep=1.0e+2, w_reg=1.0e-1, w_tmp=5.0e+1)
    img = fit(loss, spherical, obs, N_epochs=250, lr=1.0e-2, w_rep=1.0e+3, w_reg=1.0e-0, w_tmp=1.0e+2)
    
    return img, loss


def fit2(loss, spherical, obs, N_epochs=10, lr=1.0e-1, w_rep=1.0, w_reg=1.0, w_cnt=1.0, w_tmp=1.0):

    r = spherical.model_1D.get_coords(origin=np.array([-2]))
    r[r<r_in] = r_in

    optimizer = Adam(spherical.model_1D.free_parameters(), lr=lr)

    for _ in tqdm(range(N_epochs)):
        
        # Run forward model
        img = forward(spherical)
 
        # Compute the reproduction loss
        loss['rep'] = w_rep * torch.nn.functional.mse_loss(img, obs)
        # Compute the regularisation loss
        loss['reg'] = w_reg * spherical.model_1D.apply(diff_loss)
        # Compute the hydrodynamic loss
        loss['tmp'] = w_tmp * steady_state_heat_loss(spherical, torch.from_numpy(r))    
        loss['cnt'] = w_cnt * steady_state_cont_loss(spherical, torch.from_numpy(r))    

        # Set gradients to zero
        optimizer.zero_grad()
        # Backpropagate gradients
        loss.tot().backward()
        # Update parameters
        optimizer.step()

    return img


def reconstruct2(spherical, obs):

    loss = Loss(['rep', 'reg', 'tmp', 'cnt'])
    # loss = Loss(['rep', 'reg', 'tmp'])

    img = fit2(loss, spherical, obs, N_epochs=3  , lr=1.0e-1, w_rep=1.0e+0, w_reg=1.0e-0, w_tmp=1.0e+0, w_cnt=1.0e+0)
    loss.renormalise_all()
    loss.reset()
    img = fit2(loss, spherical, obs, N_epochs=550, lr=5.0e-1, w_rep=1.0e+0, w_reg=1.0e-1, w_tmp=1.0e+2, w_cnt=1.0e+0)
    img = fit2(loss, spherical, obs, N_epochs=550, lr=1.0e-1, w_rep=1.0e+2, w_reg=1.0e-0, w_tmp=5.0e+2, w_cnt=1.0e+1)
    img = fit2(loss, spherical, obs, N_epochs=550, lr=5.0e-2, w_rep=1.0e+3, w_reg=1.0e-1, w_tmp=1.0e+3, w_cnt=5.0e+1)
    img = fit2(loss, spherical, obs, N_epochs=550, lr=1.0e-2, w_rep=1.0e+5, w_reg=1.0e+2, w_tmp=5.0e+3, w_cnt=1.0e+2)
    
    return img, loss