import matplotlib.pyplot as plt
import numpy             as np
import torch
import copy

from torch.optim import Adam
from tqdm        import tqdm
from astropy     import units, constants
from pomme.model import TensorModel, SphericalModel
from pomme.loss  import Loss, diff_loss
from pomme.plot  import plot_cube_2D
from pomme.utils import planck, T_CMB

from spherical import lines, velocities, frequencies, smodel, r_out, get_velocity, get_temperature, get_turbulence, get_boundary_condition
from spherical import smodel as smodel_truth


obss = torch.load('obss.pt')


def get_abundance(model):
    """
    Get the CO abundance from the TensorModel.
    """
    # Extract parameters
    r      = torch.exp(model['log_r'])
    M_dot  = torch.exp(model['log_M_dot'])
    R_star = torch.exp(model['log_R_star'])

    v = get_velocity(model)

    rho  = M_dot / (4.0 * np.pi * r**2 * v)
    n_CO = (3.0e-4 * constants.N_A.si.value / 2.02e-3) * rho
    n_CO[r<=R_star] = n_CO[n_CO<np.inf].max()

    return n_CO


def fit(losses, smodel, lines, frequencies, obss, N_epochs=10, lr=1.0e-1, w_avg=1.0, w_rel=1.0):

    obss_avg = obss.mean(axis=1)
    obss_rel = torch.einsum("ij, i -> ij", obss, 1.0 / obss.mean(axis=1))

    params = smodel.model_1D.free_parameters()

    abundance_evol = [smodel.get_abundance(smodel.model_1D).detach().clone()]
    
    optimizer = Adam(params, lr=lr)

    for _ in tqdm(range(N_epochs)):

        # Forward model
        imgs = smodel.image(lines, frequencies, r_max=r_out)

        imgs_avg= imgs.mean(axis=1)
        imgs_rel= torch.einsum("ij, i -> ij", imgs, 1.0 / imgs.mean(axis=1))

        # Compute the reproduction loss
        losses['avg'] = w_avg * torch.nn.functional.mse_loss(imgs_avg, obss_avg)
        losses['rel'] = w_rel * torch.nn.functional.mse_loss(imgs_rel, obss_rel)

        # Set gradients to zero
        optimizer.zero_grad()
        # Backpropagate gradients
        losses.tot().backward()
        # Update parameters
        optimizer.step()

        abundance_evol.append(smodel.get_abundance(smodel.model_1D).detach().clone())

    return imgs, losses, abundance_evol


def fit_M_dot(i):

    smodel = SphericalModel(
        rs       = smodel_truth.rs,
        model_1D = TensorModel.load('model_truth.h5'),
        r_star   = smodel_truth.r_star,
    )
    smodel.get_abundance          = get_abundance
    smodel.get_velocity           = get_velocity
    smodel.get_temperature        = get_temperature
    smodel.get_turbulence         = get_turbulence
    smodel.get_boundary_condition = get_boundary_condition

    del smodel.model_1D.vars['log_CO']

    M_dot = (i * 1.0e-6 * units.M_sun / units.yr).si.value

    smodel.model_1D['log_M_dot'] = np.log(M_dot)

    smodel.model_1D.free(['log_M_dot', 'log_v_in', 'log_v_inf', 'log_beta', 'log_T_in', 'log_epsilon'])

    losses = Loss(['avg', 'rel'])


    imgs, losses, a_evol = fit(losses, smodel, lines, frequencies, obss, N_epochs=3,   lr=1.0e-1, w_avg=1.0, w_rel=1.0)
    losses.renormalise_all()
    losses.reset()
    imgs, losses, a_evol = fit(losses, smodel, lines, frequencies, obss, N_epochs=500, lr=1.0e-1, w_avg=1.0, w_rel=1.0)

    losses.plot()
    plt.savefig(f'Mdot/losses_{i}.png')

    torch.save(imgs, f'Mdot/imgs_{i}.pt')

    smodel.model_1D.save(f'Mdot/model_{i}.h5')


if __name__ == "__main__":
    
    for i in range(1, 11):
        fit_M_dot(i)