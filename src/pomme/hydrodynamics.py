import torch

from astropy import constants, units


class Hydrodynamics:

    def __init__(self, gamma, mu, f_x=0.0, f_y=0.0, f_z=0.0, Lambda=0.0):

        # Set parameters
        self.gamma  = gamma
        self.mu     = mu
        self.f_x    = f_x
        self.f_y    = f_y
        self.f_z    = f_z
        self.Lambda = Lambda

        self.loss_keys = {
            'cont': 'hydro_cont',
            'momx': 'hydro_momx',
            'momy': 'hydro_momy',
            'momz': 'hydro_momz',
            'engy': 'hydro_engy',
        }



    def add_steady_state_loss(self, model, loss, log_rho, log_tmp, v_x, v_y, v_z):
        """
        Loss assuming steady state hydrodynamics, i.e. vanishing time derivatives in Euler's equations.
        """
        rho = torch.exp(log_rho)         
        tmp = torch.exp(log_tmp)
  
        # kB T / mu
        kBT_o_mu = (constants.k_B.si.value / self.mu) * tmp
    
        # log rho + log T
        log_rho_p_log_tmp = log_rho + log_tmp

        # Energy    
        eng = 0.5 * (v_x**2 + v_y**2 + v_z**2) + (self.gamma / (self.gamma - 1.0)) * kBT_o_mu

        # Continuity equation (steady state): div(ρ v) = 0
        loss_cont = model.diff_x(rho * v_x) + model.diff_y(rho * v_y) + model.diff_z(rho * v_z)

        # Momentum equation (steady state): v . grad(v) + grad(P) / rho = f
        loss_momx = v_x * model.diff_x(v_x) + v_y * model.diff_y(v_x) + v_z * model.diff_z(v_x) + kBT_o_mu * model.diff_x(log_rho_p_log_tmp) - self.f_x
        loss_momy = v_x * model.diff_x(v_y) + v_y * model.diff_y(v_y) + v_z * model.diff_z(v_y) + kBT_o_mu * model.diff_y(log_rho_p_log_tmp) - self.f_y
        loss_momz = v_x * model.diff_x(v_z) + v_y * model.diff_y(v_z) + v_z * model.diff_z(v_z) + kBT_o_mu * model.diff_z(log_rho_p_log_tmp) - self.f_z

        # Energy equation (steady state): div(u v) = 0
        loss_engy = rho * (model.diff_x(eng) * v_x + model.diff_y(eng) * v_y + model.diff_z(eng) * v_z) - self.Lambda

        # Compute the mean squares and add the losses
        loss[self.loss_keys['cont']] = torch.mean((loss_cont/     rho )**2)
        loss[self.loss_keys['momx']] = torch.mean((loss_momx/     v_x )**2)
        loss[self.loss_keys['momy']] = torch.mean((loss_momy/     v_y )**2)
        loss[self.loss_keys['momz']] = torch.mean((loss_momz/     v_z )**2)
        loss[self.loss_keys['engy']] = torch.mean((loss_engy/(rho*eng))**2)

        # Return losses
        return loss