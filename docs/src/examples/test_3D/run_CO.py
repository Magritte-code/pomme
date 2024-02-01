import torch
import numpy as np

from torch.optim    import Adam
from tqdm           import tqdm
from astropy        import constants, units
from pomme.model import TensorModel
from pomme.loss  import Loss, diff_loss, SphericalLoss
from pomme.lines import Line


vmax = 13000.0


line = Line('CO', 2)

vdiff = 350   # velocity increment size [m/s]
nfreq = 100   # number of frequencies
dd    = vdiff / constants.c.si.value * nfreq
fmin  = line.frequency - line.frequency*dd
fmax  = line.frequency + line.frequency*dd

frequencies = torch.linspace(fmin, fmax, nfreq)
velocities = (frequencies / line.frequency - 1.0) * constants.c.si.value


def forward(model):
    
    img = line.LTE_image_along_last_axis(
        density      = torch.exp(model['log_CO'          ]),
        temperature  = torch.exp(model['log_temperature' ]),
        v_turbulence = torch.exp(model['log_v_turbulence']),
        velocity_los =    vmax * model['velocity_z'      ] ,
        frequencies  = frequencies,
        dx           = model.dx(0)
    )
    
    return 1.0e+12 * img


def analytic_velo(r, r_in, v_in, v_inf, beta):
    return v_in + (v_inf - v_in) * (1.0 - r_in / r)**beta

def analytic_T(r, r_in, T_in, epsilon):
    return T_in * (r_in / r)**epsilon


def forward_analytic_velo_and_T(model):

    r    = model.get_radius(origin='centre')
    r_in = r.min()
    d    = model.get_radial_direction(origin='centre')[2]

    velocity_r = analytic_velo(
        r     = torch.from_numpy(r),
        r_in  = r_in,
        v_in  = torch.exp(model['log_v_in']),
        v_inf = torch.exp(model['log_v_inf']),
        beta  =           model['beta']
    )
    
    temperature = analytic_T(
        r       = torch.from_numpy(r),
        r_in    = r_in,
        T_in    = torch.exp(model['log_T_in']),
        epsilon = torch.exp(model['log_epsilon'])
    )

    img = line.LTE_image_along_last_axis(
        abundance    = torch.exp(model['log_CO'          ]),
        temperature  = temperature,
        v_turbulence = torch.exp(model['log_v_turbulence']),
        velocity_los = velocity_r * d,
        frequencies  = frequencies,
        dx           = model.dx(0)
    )
    
    return 1.0e+12 * img


def steady_state_continuity_loss(model, rho, v_x, v_y, v_z):
    """
    Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives in Euler's equations.
    """
    # Continuity equation (steady state): div(ρ v) = 0
    loss_cont = model.diff_x(rho * v_x) + model.diff_y(rho * v_y) + model.diff_z(rho * v_z)
    # Squared average over the model
    loss_cont = torch.mean((loss_cont / rho)**2)

    return loss_cont


def poslog(x):
    x_min_allowed = x.max() * 1.0e-10 
    x_min         = x.min()
    return torch.log(x + (x_min_allowed - x_min))


def fit(loss, model, obs, N_epochs=100, lr=1.0e-1, w_rep=1.0, w_nrm_rep=1.0, w_reg=1.0, w_cnt=1.0, w_sph=1.0):

    params = [
        model['log_CO'],
        model['log_v_in'],
        # model['log_v_inf'],
        model['beta'],
        model['log_T_in'],
        model['log_epsilon'],
    ]

    optimizer = Adam(params, lr=lr)

    sphericalLoss = SphericalLoss(model, origin='centre')

    for _ in tqdm(range(N_epochs)):
        
        # Run forward model
        img = forward_analytic_velo_and_T(model)
 
        # Compute the reproduction loss
        # loss['log_rep'] = w_log_rep * torch.nn.functional.mse_loss(poslog(img), poslog(obs))

        loss['rep'] = w_rep * torch.nn.functional.mse_loss(img, obs)

        nrm = 1.0 / obs.mean(dim=2)

        loss['nrm_rep'] = w_nrm_rep * torch.nn.functional.mse_loss(
            torch.einsum('ijf, ij -> ijf', img, nrm),
            torch.einsum('ijf, ij -> ijf', obs, nrm)
        )

        # Compute the regularisation loss
        loss['reg'] = w_reg * diff_loss(model['log_CO'])

        # loss['sph'] = w_sph * sphericalLoss.eval(torch.exp(model['log_CO']))

        r    = model.get_radius(origin='centre')
        r_in = r.min()
        d    = model.get_radial_direction(origin='centre')

        velocity_r = analytic_velo(
            r     = torch.from_numpy(r),
            r_in  = r_in,
            v_in  = torch.exp(model['log_v_in']),
            v_inf = torch.exp(model['log_v_inf']),
            beta  =           model['beta']
        )

        # Compute the hydrodynamic loss
        loss['cnt'] = w_cnt * steady_state_continuity_loss(
            model = model,
            rho   = torch.exp(model['log_CO']),
            v_x   = velocity_r * d[0],
            v_y   = velocity_r * d[1],
            v_z   = velocity_r * d[2],
        )      

        # Set gradients to zero
        optimizer.zero_grad()
        # Backpropagate gradients
        loss.tot().backward()
        # Update parameters
        optimizer.step()

        # print('loss_tot =', loss.tot().item())

    return img

    
model = TensorModel.load('models/model_3D.h5')

obs = forward(model)
obs = np.abs(obs) + 1.0e-20


def init():
    # r_x, r_y, r_z = model.get_radial_direction(origin='centre')

    model = TensorModel.load('models/model_3D.h5')

    model.vars.pop('log_temperature')
    model.vars.pop('velocity_x')
    model.vars.pop('velocity_y')
    model.vars.pop('velocity_z')

    r = model.get_radius(origin='centre')
    d = r / r.min()

    model['log_CO'          ] = np.log(1.0e+10 * (1.0/d)**2)
    model['log_v_in'        ] = np.log(1.0e+1)
    model['log_v_inf'       ] = np.log(vmax)
    model['beta'            ] = 1.0
    model['log_T_in'        ] = np.log(5.0e+3)
    model['log_epsilon'     ] = np.log(0.5)


    # model['log_temperature' ] = np.log(1.0e+3) * np.ones(model.shape)
    # model['log_v_turbulence'] = np.log(1.5e+2) * np.ones(model.shape)
    # model['velocity_x'      ] = 1.0e+0 * r_x
    # model['velocity_y'      ] = 1.0e+0 * r_y
    # model['velocity_z'      ] = 1.0e+0 * r_z

    # Explicitly set all model variables free (i.e. all will be fitted) except the turbulence
    model.fix_all()
    model.free('log_CO' )
    model.free('log_v_in')
    # model.free('log_v_inf')
    model.free('beta')
    model.free('log_T_in')
    model.free('log_epsilon')
    # model.fix('log_v_turbulence')
    # model.fix('velocity_x')
    # model.fix('velocity_y')
    # model.fix('velocity_z')

    # loss = Loss(['rep', 'reg', 'cnt', 'tmp'])
    # loss = Loss(['rep', 'reg', 'cnt'])
    loss = Loss(['rep', 'nrm_rep', 'reg', 'cnt'])
    # loss = Loss(['log_rep', 'reg', 'sph'])

    img = forward_analytic_velo_and_T(model)

    # img = fit(loss, model, obs, N_epochs=3,   lr=5.0e-1, w_rep=1.0e+2)
    # loss.renormalise_all()
    # loss.reset()

    model.save('models/model_3D_CO_all.h5')

    return img, loss


def run():

    loss = Loss(['rep', 'reg', 'cnt'])

    model = TensorModel.load('models/model_3D_CO_all.h5')

    img = fit(loss, model, obs, N_epochs=3,   lr=5.0e-1, w_rep=1.0e+2)
    img = fit(loss, model, obs, N_epochs=100, lr=5.0e-1, w_rep=1.0e+2)

    model.save('models/model_3D_CO_all.h5')

    return img, loss


if __name__ == '__main__':
    run()