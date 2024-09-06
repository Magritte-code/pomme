import healpy as hp
import torch
import math
import gc
import numpy as np
import matplotlib.pyplot as plt
from astropy        import units, constants
import astropy.constants as constants
from p3droslo.plot  import plot_cube_2D, plot_spectrum
from p3droslo.model import TensorModel
from p3droslo.lines import Line
from p3droslo.haar  import Haar
import copy
import p3droslo
print(p3droslo.__version__)
from astroquery.lamda import Lamda

CC  = constants.c  .si.value   # Speed of light       [m/s]
HH  = constants.h  .si.value   # Planck's constant    [J s]
KB  = constants.k_B.si.value   # Boltzmann's constant [J/K]
AMU = constants.u  .si.value   # Atomic mass unit     [kg]


def compute_sigma(line,temperature,v_turbulence):

    factor_1 = line.frequency / CC
    factor_2 = 2.0 * KB / (line.species_molar_mass * AMU)
    return factor_1 * torch.sqrt(factor_2*temperature + v_turbulence**2)
def compute_profile(frequencies,linefreq,sigma):
    #print(frequencies.shape)
    profile=torch.zeros(*sigma.shape,len(frequencies))
    #while frequencies.dim() < linefreq.dim()+1:
    #    frequencies = frequencies.unsqueeze(0)
    mask=sigma>1e-20
    profile[mask]=(1/(np.sqrt(np.pi)*sigma[...,None][mask]))*torch.exp(-(frequencies[None,None,None,...]-linefreq[...,None][mask])**2/(sigma[...,None][mask]**2))
    #profile=(1/(torch.pi*sigma[...,None]))*torch.exp(-(frequencies[None,None,None,:]-linefreq)**2/(sigma[...,None]**2))
    return profile.float()
def source_ij(line,pop):
    """
    Line source function

    Parameters
    ----------
    pop : torch.Tensor
        level populations
    
    Returns
    -------
    src : torch.Tensor
    Tensor containing the line source function (= emissivity / opacity).
    """
    pop=pop.to(torch.float32)
    upper_mask=pop[line.upper,:]>1e-20 
    lower_mask=pop[line.lower,:]>1e-20
    mask=upper_mask&lower_mask
    source=torch.zeros(mask.shape)
    source[mask]=(line.Einstein_A  * pop[line.upper][mask] / (line.Einstein_Ba * pop[line.lower][mask] - line.Einstein_Bs * pop[line.upper][mask]) )
    return source
def emissivity_and_opacity_ij(line,model, pop):
    """
    Line opacity, not folded with the profile.
        
    Parameters
    ----------
    pop : torch.Tensor
        level populations
    
    Returns
    -------
    eta, chi : torch.Tensor, torch.Tensor
    Tensor containing the LTE emissivities and opacities for the given temperature.
    """
    upper_mask=pop[line.upper,:]>1e-30 
    lower_mask=pop[line.lower,:]>1e-30
    mask=upper_mask&lower_mask
    eta=torch.zeros(model.shape)
    chi=torch.zeros(model.shape)
    # Compute the prefactor
    factor = HH * line.frequency / (4.0 * np.pi)
        
    # Compute the emissivity and opacity
    eta[mask] = factor *  line.Einstein_A  * pop[line.upper][mask]
    chi[mask] = factor * (line.Einstein_Ba * pop[line.lower][mask] - line.Einstein_Bs * pop[line.upper][mask]) 
        
    # Return results
    return eta, chi

def optical_depth_along_last_axis( line,chi_ij, sigma,abundance, temperature, velocity_los, frequencies, dx):
        """
        Line optical depth along the last axis.
        """      
        sqrt_pi = np.sqrt(np.pi)
        temp_mask=temperature>1e-10
        inverse_width =torch.zeros_like(sigma)
        # Compute inverse line width
        inverse_width[temp_mask] = 1.0 / sigma[temp_mask]
        # Get the index of the last spatial axis
        last_axis = abundance.dim() - 1
        # Compute the Doppler shift for each element
        shift = 1.0 + velocity_los * (1.0 / CC)
        # Create freqency tensor in the rest frame for each element
        freqs_restframe = torch.einsum("..., f -> ...f", shift, frequencies)
        # Define the a and b (tabulated) functions (Note: we absorb dx in the definition of a for efficiency)
        a = (1.0/sqrt_pi) * inverse_width * chi_ij * abundance
        b = torch.einsum("..., ...f -> ...f", inverse_width, freqs_restframe - line.frequency)
        #if a.isnan().any():
        #    raise Warning("NaNs in a.")
        #if b.isnan().any():
        #    raise Warning("NaNs in b.")
        expb = torch.exp(-b**2) 
        erfb = torch.erf( b   ) 
        a0 = a[..., :-1] * dx
        a1 = a[..., 1: ] * dx
        b0 = b[..., :-1, :]
        b1 = b[..., 1: , :]
        expb0 = expb[..., :-1, :]
        expb1 = expb[..., 1: , :]
        erfb0 = erfb[..., :-1, :]
        erfb1 = erfb[..., 1: , :]
        b10 = b1 - b0
        a01 = a0 + a1
        sp_a1b0 = torch.einsum("..., ...f -> ...f", sqrt_pi * a1, b0)
        sp_a0b1 = torch.einsum("..., ...f -> ...f", sqrt_pi * a0, b1)
        #if a1.isnan().any():
        #     raise Warning("NaNs in a1.")
        #if a0.isnan().any():
        #     raise Warning("NaNs in a0.")
        #if b0.isnan().any():
        #     raise Warning("NaNs in b0.")
        #if b1.isnan().any():
        #     raise Warning("NaNs in b1.")
        #if sp_a1b0.isnan().any():
        #     raise Warning("NaNs in sp_a1b0.")
        #if sp_a0b1.isnan().any():
        #     raise Warning("NaNs in sp_a0b1.")
        dtau  = torch.einsum("..., ...f -> ...f", a1 - a0, expb0 - expb1)
        dtau += (sp_a1b0 - sp_a0b1) * (erfb0 - erfb1)
        dtau *= (0.5 / (b10**2 + 1.0e-30))
        # note that the regularizer 1.0e-30 is never going to be significant
        # however it is essential to avoid nans in backprop (see https://github.com/Magritte-code/pomme/issues/2)
        #if dtau.isnan().any():
        #     raise Warning("NaNs in dtau before mask.")
        # Create a mask for the region where the above calculation might have numerical issues
        mask_threshold = 1.0e-4
        mask           = torch.Tensor(torch.abs(b10) < mask_threshold)
        # Use a (second order) Taylor expansion in b10 for the masked region
        dtau[mask] = (torch.einsum("..., ...f", (1.0/2.0) *  a01      , expb0           ))[mask]
                      # - torch.einsum("..., ...f", (1.0/3.0) * (a01 + a1), expb0 * b0 * b10) )[mask]
        # if dtau.isnan().any():
        #     raise Warning("NaNs in dtau after mask.")
        tau = torch.empty_like(b)
        tau[...,  0 , :] = 0.0
        tau[..., +1:, :] = torch.cumsum(dtau, dim=last_axis)
        return dtau, tau

def image_along_last_axis(src, dtau, tau):
    # Check dimensionality of the input
    assert src.dim() == tau.dim()-1, "Tensor src should only have spatial dimensions, no frequency!"
    # Get the index of the last spatial axis
    last_axis = src.dim() - 1
    #dtau=torch.diff(tau,dim=last_axis)
    src_0 = src[..., :-1]
    src_1 = src[..., +1:]

    exp_minus_tau = torch.exp(-tau)

    #emt_0 = exp_minus_tau[..., :-1, :]
    #emt_1 = exp_minus_tau[..., +1:, :]
    emt_0 = torch.ones_like(dtau)
    emt_1 = torch.exp(-dtau)
    # threshold differentiating the two optical dpeth regimes
    mask_threshold = 1.0e-4
    #mask = torch.Tensor(dtau < mask_threshold)
    mask = dtau < mask_threshold
    # Case a: dtau > threshold 
    result  = torch.einsum("..., ...f -> ...f", src_0, emt_1 - emt_0 * (1.0 - dtau))
    result += torch.einsum("..., ...f -> ...f", src_1, emt_0 - emt_1 * (1.0 + dtau))
    result /= (dtau + 1.0e-30)

    # note that the regularizer 1.0e-30 is never going to be significant
    # however it is essential to avoid nans in backprop (see https://github.com/Magritte-code/p3droslo/issues/2)
    # Use a Taylor expansion for small dtau
    cc     = (1.0/2.0) * dtau
    fac_0  = cc.clone() 
    fac_1  = cc.clone()
    cc    = (1.0/3.0) * dtau *cc
    fac_0 = fac_0+cc 
    fac_1 = fac_1-cc
    cc    = (1.0/4.0) * dtau *cc
    fac_0 = fac_0+cc
    fac_1 = fac_1-cc
    result[mask] = (torch.einsum("..., ...f -> ...f", src_0, emt_0 * fac_0) \
                    + torch.einsum("..., ...f -> ...f", src_1, emt_1 * fac_1) )[mask]

    #img=torch.zeros_like(tau)
    #img[...,0,:]=src[...,0].unsqueeze(-1)*(1-torch.exp(tau[...,0,:]))
    specific_intensity=(src[...,0].unsqueeze(-1)*(1-torch.exp(tau[...,0,:]))).unsqueeze(last_axis)
    #print(specific_intensity.shape)
    #for i in range(1,img.shape[2]):
    #    img[...,i,:]=result[...,i-1,:]+img[...,i-1,:]*torch.exp(-dtau[...,i-1,:])
    for i in range(1,tau.shape[2]):
        img_slice = result[..., i - 1, :] + specific_intensity[..., i - 1, :] * torch.exp(-dtau[..., i - 1, :])
        specific_intensity=torch.cat((specific_intensity,img_slice.unsqueeze(last_axis)),dim=last_axis)
        #img[..., i, :] = img_slice

    #specific_intensity=img

     
    return specific_intensity
    
    
def compute_SE(model,line,C_ij,C_ji,J,NLTE_pops):
    dni_dt=torch.zeros(len(line.linedata.level),*model.shape)
    for i in range(1,len(line.linedata.level)):
        j=i-1
        sum_Pji_nj=0
        sum_Pij_ni=0
        
        n_i=(model['density'] * NLTE_pops[i,...])
        n_j=(model['density']* NLTE_pops[j,...])
        A_ij=line.linedata.Einstein_A[i-1]
        B_ji=line.linedata.Einstein_Bs[i-1]
        B_ij=line.linedata.Einstein_Ba[i-1]
        R_ij=A_ij+B_ij*J
        R_ji=B_ji*J  

        P_ij=transition_Rate(R_ij,C_ij)
        P_ji=transition_Rate(R_ji,C_ji)
        sum_Pji_nj=n_j*P_ji
        sum_Pij_ni=n_i*P_ij
        
        dni_dt[i,...]=sum_Pji_nj-sum_Pij_ni   
        dni_dt[j,...]=sum_Pij_ni-sum_Pji_nj   
    return dni_dt
  
    
