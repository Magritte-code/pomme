import torch

from astropy import constants, units


class Hydrodynamics:

    def __setup__(self, gamma=1.2, mu=2.381*constants.u.si.value):
        self.gamma = gamma
        self.mu    = mu




gamma = 1.2
mu    = 2.381 * constants.u.si.value

def steady_state_hydrodynamic_loss(model, f_x=0.0, f_y=0.0, f_z=0.0, heating_m_cooling=0.0):
    """
    Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives in Euler's equations.
    """

    r = model.get_coords(origin=origin_ind)
    d = torch.from_numpy(r / np.linalg.norm(r, axis=0)**3)
    
    log_rho = model['log_CO']
    log_tmp = model['log_temperature'] 
    log_M   = model['log_M']

    rho = torch.exp(log_rho)         
    tmp = torch.exp(log_tmp)
    M   = torch.exp(log_M)
    
    v_x = v_max * model['velocity_x']
    v_y = v_max * model['velocity_y']
    v_z = v_max * model['velocity_z']

    kBT_o_mu = (constants.k_B.si.value / mu) * tmp

    # Energy    
    eng = 0.5 * (v_x**2 + v_y**2 + v_z**2) + (gamma / (gamma - 1.0)) * kBT_o_mu

    # log rho + log T
    log_rho_p_log_tmp = log_rho + log_tmp
    
    f_x = -constants.G.si.value * M * d[0]
    f_y = -constants.G.si.value * M * d[1]
    f_x = -constants.G.si.value * M * d[2]

    # Continuity equation (steady state): div(ρ v) = 0
    loss_cont = model.diff_x(rho * v_x) + model.diff_y(rho * v_y) + model.diff_z(rho * v_z)

    # Momentum equation (steady state): v . grad(v) + grad(P) / rho = f
    loss_momx = v_x * model.diff_x(v_x) + v_y * model.diff_y(v_x) + v_z * model.diff_z(v_x) + kBT_o_mu * model.diff_x(log_rho_p_log_tmp) - f_x
    loss_momy = v_x * model.diff_x(v_y) + v_y * model.diff_y(v_y) + v_z * model.diff_z(v_y) + kBT_o_mu * model.diff_y(log_rho_p_log_tmp) - f_y
    loss_momz = v_x * model.diff_x(v_z) + v_y * model.diff_y(v_z) + v_z * model.diff_z(v_z) + kBT_o_mu * model.diff_z(log_rho_p_log_tmp) - f_z

    # Energy equation (steady state): div(u v) = 0
    loss_engy = rho * (model.diff_x(eng) * v_x + model.diff_y(eng) * v_y + model.diff_z(eng) * v_z) - heating_m_cooling

    # Compute the mean squared losses
    losses = torch.stack([
        ((loss_cont/     rho )**2).mean(),
        ((loss_momx/     v_x )**2).mean(),
        ((loss_momy/     v_y )**2).mean(),
        ((loss_momz/     v_z )**2).mean(),
        ((loss_engy/(rho*eng))**2).mean()
    ])

    # Return losses
    return losses

    # def loss_cont(self, )