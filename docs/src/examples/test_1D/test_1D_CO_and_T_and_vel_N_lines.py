import matplotlib.pyplot as plt
import numpy             as np
import torch

from torch.optim          import Adam
from tqdm                 import tqdm
from astropy              import units, constants

from pomme.model       import TensorModel, SphericallySymmetric
from pomme.lines       import Line
from pomme.loss        import Loss, diff_loss

from test_1D import forward, forward_N_lines, forward_analytic_velo, forward_analytic_velo_and_T, forward_analytic_velo_and_T_N_lines, get_model, line, frequencies, velocities, r_in, v_fac


def get_initial_model(from_model, nCO, v_in, v_inf, beta, tmp, epsilon):

    # Define and initialise the model variables
    model_1D = TensorModel.load(from_model)
    model_1D['log_CO'         ] = np.log(nCO) * np.ones(model_1D.shape)
    # model_1D['log_velocity'   ] = np.log(1/v_fac * vel) * np.ones(model_1D.shape)
    # model_1D['log_temperature'] = np.log(tmp) * np.ones(model_1D.shape)

    model_1D['log_v_in'   ] = np.log(v_in)
    model_1D['log_v_inf'  ] = np.log(v_inf)
    model_1D['log_beta'   ] = np.log(beta)

    model_1D['log_T_in'   ] = np.log(tmp)
    model_1D['log_epsilon'] = np.log(epsilon)

    model_1D.fix_all()
    model_1D.free('log_CO')
    model_1D.free('log_v_in')
    model_1D.free('log_v_inf')
    model_1D.free('log_beta')
    model_1D.free('log_T_in')
    model_1D.free('log_epsilon')
    # model_1D.free('log_velocity')
    # model_1D.free('log_temperature')

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

    # Compute the mean squared losses
    loss = torch.mean(loss_heat**2)

    # Return losses
    return loss


def fit_N_lines(losses, spherical, lines, frequencies, obss, N_epochs=10, lr=1.0e-1, w_rep=1.0, w_reg=1.0, w_cnt=1.0):

    r = spherical.model_1D.get_coords(origin=np.array([0]))
    r[r<r_in] = r_in

    params = [
        spherical.model_1D['log_CO'],
        spherical.model_1D['log_v_in'],
        spherical.model_1D['log_v_inf'],
        spherical.model_1D['log_beta'],
        spherical.model_1D['log_T_in'],
        spherical.model_1D['log_epsilon'],
    ]

    optimizer = Adam(params, lr=lr)

    for _ in tqdm(range(N_epochs)):
                 
        for i, (line, freq, obs) in enumerate(zip(lines, frequencies, obss)):
            # Run forward model
            img = forward_analytic_velo_and_T(spherical, line, freq)

            # Compute the reproduction loss
            losses[i]['rep'] = w_rep * torch.nn.functional.mse_loss(img, obs)
            # Compute the regularisation loss
            losses[i]['reg'] = w_reg * diff_loss(spherical.model_1D['log_CO'])
            # Compute the hydrodynamic loss
            # loss['tmp'] = w_tmp * steady_state_heat_loss(spherical, torch.from_numpy(r))    
            losses[i]['cnt'] = w_cnt * steady_state_cont_loss(spherical, torch.from_numpy(r))    

            # Set gradients to zero
            optimizer.zero_grad()
            # Backpropagate gradients
            losses[i].tot().backward()
            # Update parameters
            optimizer.step()

    return img, losses


def reconstruct_N_lines(spherical, lines, frequencies, obss):
    """
    Reconsturction recipe.
    """
    loss_keys  = [f'rep_{i}' for i in range(len(obss))]
    loss_keys += ['reg', 'cnt']
    loss = Loss(loss_keys)

    imgs, loss = fit_N_lines(loss, spherical, lines, frequencies, obss, N_epochs=3  , lr=1.0e-1, w_rep=1.0e+0, w_reg=1.0e-0, w_cnt=1.0e+0)
    loss.renormalise_all()
    loss.reset()
    imgs, loss = fit_N_lines(loss, spherical, lines, frequencies, obss, N_epochs=250, lr=1.0e-1, w_rep=1.0e+0, w_reg=1.0e-1, w_cnt=1.0e+0)
    # imgs, loss = fit_N_lines(loss, spherical, lines, frequencies, obss, N_epochs=250, lr=1.0e-1, w_rep=1.0e+2, w_reg=1.0e+0, w_cnt=1.0e+1)
    # imgs, loss = fit_N_lines(loss, spherical, lines, frequencies, obss, N_epochs=250, lr=1.0e-1, w_rep=1.0e+3, w_reg=1.0e+0, w_cnt=5.0e+1)
    # imgs, loss = fit_N_lines(loss, spherical, lines, frequencies, obss, N_epochs=250, lr=1.0e-1, w_rep=1.0e+5, w_reg=1.0e+0, w_cnt=1.0e+2)
    
    return imgs, loss