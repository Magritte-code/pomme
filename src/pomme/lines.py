import torch
import numpy as np

from time             import time
from astroquery.lamda import Lamda, parse_lamda_datafile
from astropy          import constants, units
from pomme.utils   import get_molar_mass, print_var
from pomme.forward import image_along_last_axis as forward_image_along_last_axis


# Constants
CC  = constants.c  .si.value   # Speed of light       [m/s]
HH  = constants.h  .si.value   # Planck's constant    [J s]
KB  = constants.k_B.si.value   # Boltzmann's constant [J/K]
AMU = constants.u  .si.value   # Atomic mass unit     [kg]


class LineData():
    """
    Class structuring the line data.
    """
    
    def __init__(self, species_name=None, datafile=None, database='LAMDA'):
        """
        Constructor for linedata object.
        Either provide a datafile or a molecule name.
        
        Parameters
        ----------
        species_name : str
            Name of the lineproducing species.
        database : str
            Name of the type of line data base. (Default LAMDA.)
        """
        
        if database == 'LAMDA':
            # Use astroquery to read the LAMDA file
            if datafile is not None:
                collrates, radtransitions, enlevels = parse_lamda_datafile(datafile)
            elif species_name is not None:
                collrates, radtransitions, enlevels = Lamda.query(mol=species_name)
            else:
                raise ValueError("Either datafile or mol must be specified.")
        
            self.level  = np.array(enlevels['Level' ])
            self.energy = np.array(enlevels['Energy'])
            self.weight = np.array(enlevels['Weight'])
            self.J      = np.array(enlevels['J'     ])
            
            self.transition = np.array(radtransitions['Transition'])
            self.upper      = np.array(radtransitions['Upper'     ])
            self.lower      = np.array(radtransitions['Lower'     ])
            self.Einstein_A = np.array(radtransitions['EinsteinA' ])
            self.frequency  = np.array(radtransitions['Frequency' ])
            
            # Unit conversions to SI
            self.energy    *= 1.0E+2*HH*CC   # from [cm^-1] to [J]
            self.frequency *= 1.0e+9         # from [GHz]   to [Hz] 
            
            # Renumber levels and transitions from [1 - ] to [0 - ]
            self.level      -= 1
            self.transition -= 1
            self.upper      -= 1
            self.lower      -= 1
            
            # Derive Einstein B coefficients
            self.Einstein_Bs = self.Einstein_A * CC**2 / (2.0*HH*(self.frequency)**3)
            self.Einstein_Ba = self.weight[self.upper] / self.weight[self.lower] * self.Einstein_Bs
            
        else:
            raise NotImplementedError("Currently, only LAMDA data files are supported.")
            
            
class Line:
    """
    Spectral line class.
    """
    
    def __init__(self, species_name, transition, database='LAMDA', datafile=None, molar_mass=None):
        """
        Constructor for a line object.

        Parameters
        ----------
        species_name : str
            Name of the line producing species.
        transition : int
            Transition number of the line.
        database : str
            Name of the type of line data base. (Default LAMDA.)
        datafile : str
            Path to the datafile. (Default None.)
        molar_mass : float
            Molar mass of the species. (Default None.)
        """
        # Store the name of the species
        self.species_name = species_name
        # Determine the molar mass of the species
        if molar_mass is None:
            self.species_molar_mass = get_molar_mass(self.species_name)
        else:
            self.species_molar_mass = molar_mass
        # Store the transition number
        self.transition = transition
        # Extract the line data
        self.linedata = LineData(database=database, datafile=datafile, species_name=species_name.lower())
        
        self.upper       = self.linedata.upper       [self.transition]
        self.lower       = self.linedata.lower       [self.transition]
        self.frequency   = self.linedata.frequency   [self.transition]
        self.Einstein_A  = self.linedata.Einstein_A  [self.transition]
        self.Einstein_Bs = self.linedata.Einstein_Bs [self.transition]
        self.Einstein_Ba = self.linedata.Einstein_Ba [self.transition]
        
        self.J_upper = self.linedata.J[self.upper]
        self.J_lower = self.linedata.J[self.lower]
        
        self.energy = torch.from_numpy(self.linedata.energy)
        self.weight = torch.from_numpy(self.linedata.weight)
        
        self.description = f"{self.species_name}(J={self.J_upper}-{self.J_lower})"

        print(f"You have selected line:")
        print(f"    {self.description}")
        print(f"Please check the properties that were inferred:")
        print(f"    {'Frequency '       :<17} {self.frequency :0.9e}  Hz")
        print(f"    {'Einstein A coeff ':<17} {self.Einstein_A:0.9e}  1/s")
        print(f"    {'Molar mass'       :<17} {self.species_molar_mass:<15}  g/mol")

        
    def gaussian_width(self, temperature, v_turbulence):
        """
        Gaussian spectral line width.

        Parameters
        ----------
        temperature : torch.Tensor
            Temperature for which to evaluate the line width.
        v_turbulence : torch.Tensor
            Turbulent velocity for which to evaluate the line width.

        Returns
        -------
        out : torch.Tensor
            Tensor containing the Gaussian line width for the given temperature and turbulent velocity
        """
        # Compute convenience variables
        factor_1 = self.frequency / CC
        factor_2 = 2.0 * KB / (self.species_molar_mass * AMU)
        # Return the gaussian line width
        return factor_1 * torch.sqrt(factor_2*temperature + v_turbulence**2)


    def gaussian_profile(self, temperature, v_turbulence, freq):
        """
        Gaussian spectral line profile function.

        Parameters
        ----------
        temperature : torch.Tensor
            Temperature for which to evaluate the line profile.
        v_turbulence : torch.Tensor
            Turbulent velocity for which to evaluate the line profile.
        freq : torch.Tensor
            Frequency at which to evaluate the line profile.

        Returns
        -------
        out : torch.Tensor
            Tensor containing the Gaussian line profile for the given temperature and turbulent velocity
        """
        # Compute convenience variables
        inverse_width = 1.0 / self.gaussian_width(temperature, v_turbulence)
        factor        = 1.0 / np.sqrt(np.pi)
        # Mind the cellwise products
        shift  = torch.einsum("..., ...f -> ...f",          inverse_width, freq-self.frequency)
        result = torch.einsum("..., ...f -> ...f", factor * inverse_width, torch.exp(-shift**2))
        # Return the gaussian line profile
        return result
    
    
    def LTE_pops (self, temperature):
        """
        Relative LTE level populations for the given temperature.
    
        Parameters
        ----------
        temperature : torch.Tensor
            Temperature for which to evaluate the LTE level populations.
    
        Returns
        -------
        out : array_like
            Array containing the LTE level populations for the given temperature.
        """
        exponent = torch.einsum("i   ,  ... -> i...", -self.energy/KB, 1.0/temperature) 
        pop      = torch.einsum("i   , i... -> i...",  self.weight,    torch.exp(exponent))
        pop      = torch.einsum("i...,  ... -> i...",  pop,            1.0/torch.sum(pop, dim=0))
        # Return result
        return pop
    

    def emissivity_and_opacity_ij(self, pop):
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
        # Compute the prefactor
        factor = HH * self.frequency / (4.0 * np.pi)
        
        # Compute the emissivity and opacity
        eta = factor *  self.Einstein_A  * pop[self.upper]
        chi = factor * (self.Einstein_Ba * pop[self.lower] - self.Einstein_Bs * pop[self.upper]) 
        
        # Return results
        return eta, chi


    def source_ij(self, pop):
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
        return self.Einstein_A  * pop[self.upper] / (self.Einstein_Ba * pop[self.lower] - self.Einstein_Bs * pop[self.upper]) 

    

    def LTE_emissivity_and_opacity(self, abundance, temperature, v_turbulence, frequencies):
        """
        Line emissivity and opacity assuming LTE.
        
        Parameters
        ----------
        abundance : torch.Tensor
            Abundance distribution of the line producing species.
        temperature : torch.Tensor
            Temperature for which to evaluate the LTE level populations.
        v_turbulence : torch.Tensor
            Turbulent velocity for which to evaluate the line profile.
        frequencies : torch.Tensor
            Frequencies at which to evaluate the LTE emissivities and opacities.
    
        Returns
        -------
        eta, chi : torch.Tensor, torch.Tensor
            Tensor containing the LTE emissivities and opacities for the given temperature.
        """
        # Compute the prefactor
        factor = HH * self.frequency / (4.0 * np.pi)
        
        # t =- time()
        # Compute the LTE level populations
        pop = self.LTE_pops(temperature)
        # t += time()
        # print("LTE pop  ", t)
        
        
        # t =- time()
        # Compute the emissivity and opacity
        eta = factor *  self.Einstein_A  * pop[self.upper]
        chi = factor * (self.Einstein_Ba * pop[self.lower] - self.Einstein_Bs * pop[self.upper]) 
        # t += time()
        # print("Eins A B ", t)
        
        # t =- time()
        # Compute the (Gaussian) line profile
        profile = self.gaussian_profile(temperature, v_turbulence, frequencies)
        # t += time()
        # print("profile  ", t)
        
        # t =- time()
        # Fold the emessivities and opacities with the profile and the number abundance
        eta = torch.einsum("..., ...f -> ...f", eta*abundance, profile)
        chi = torch.einsum("..., ...f -> ...f", chi*abundance, profile)
        # t += time()
        # print("multiply ", t)
        
        # Return results
        return (eta, chi)
    

    # def optical_depth_along_last_axis_slow(self, chi_ij, abundance, temperature, v_turbulence, velocity_los, frequencies, dx):
    #     """
    #     Line optical depth along the last axis.
    #     """
    #     sqrt_pi = np.sqrt(np.pi)

    #     # Compute inverse line width
    #     inverse_width = 1.0 / self.gaussian_width(temperature=temperature, v_turbulence=v_turbulence)
  
    #     # Get the index of the last spatial axis
    #     last_axis = abundance.dim() - 1

    #     # Compute the Doppler shift for each element
    #     shift = 1.0 + velocity_los * (1.0 / CC)

    #     # Create freqency tensor in the rest frame for each element
    #     freqs_restframe = torch.einsum("..., f -> ...f", shift, frequencies)

    #     # Define the a and b (tabulated) functions
    #     a = (1.0/sqrt_pi) * inverse_width * chi_ij * abundance
    #     a = torch.einsum("...,    f -> ...f", a, torch.ones_like(frequencies))
    #     b = torch.einsum("..., ...f -> ...f", inverse_width, freqs_restframe - self.frequency)

    #     a0 = a[..., :-1, :]
    #     a1 = a[..., 1: , :]
    
    #     b0 = b[..., :-1, :]
    #     b1 = b[..., 1: , :]

    #     b10 = b1 - b0

    #     # threashhold differentiating the two regimes (large and small Doppler shift)
    #     shift_threshold = 1.0e-3
    
    #     # Define the masks for the threashold    
    #     A = torch.Tensor(torch.abs(b10) >  shift_threshold)
    #     B = torch.Tensor(torch.abs(b10) <= shift_threshold)

    #     dtau = torch.empty_like(b10)
        
    #     dtau[A]  =           (      a1[A] -       a0[A]) * (torch.exp(-b0[A]**2) - torch.exp(-b1[A]**2))
    #     dtau[A] += sqrt_pi * (b0[A]*a1[A] - b1[A]*a0[A]) * (torch.erf( b0[A]   ) - torch.erf( b1[A]   ))
    #     dtau[A] *= 0.5 / b10[A]**2

    #     dtau[B]  = (1.0/ 2.0) * (a0[B] +     a1[B])
    #     dtau[B] -= (1.0/ 3.0) * (a0[B] + 2.0*a1[B]) * b0[B]                        * b10[B]   
    #     dtau[B] += (1.0/12.0) * (a0[B] + 3.0*a1[B]) *         (2.0*b0[B]**2 - 1.0) * b10[B]**2
    #     dtau[B] -= (1.0/30.0) * (a0[B] + 4.0*a1[B]) * b0[B] * (2.0*b0[B]**2 - 3.0) * b10[B]**3
    #     dtau[B] *= torch.exp(-b0[B]**2)
        
    #     dtau *= dx

    #     tau = torch.empty_like(b)
    #     tau[...,  0 , :] = 0.0
    #     tau[..., +1:, :] = torch.cumsum(dtau, dim=last_axis)

    #     return dtau, tau


    def optical_depth_along_last_axis(self, chi_ij, abundance, temperature, v_turbulence, velocity_los, frequencies, dx):
        """
        Line optical depth along the last axis.

        Parameters
        ----------
        chi_ij : torch.Tensor
            Line opacity distribution.
        abundance : torch.Tensor
            Abundance distribution of the line producing species.
        temperature : torch.Tensor
            Temperature distribution of the line producing species.
        v_turbulence : torch.Tensor
            Turbulent velocity distribution.
        velocity_los : torch.Tensor
            Line of sight velocity distribution.
        frequencies : torch.Tensor
            Frequencies at which to compute the optical depth.
        dx : torch.Tensor
            Grid spacing along the line of sight.

        Returns
        -------
        dtau : torch.Tensor
            Tensor containing the differential optical depth.
        tau : torch.Tensor
            Tensor containing the cumulative optical depth.
        """
        sqrt_pi = np.sqrt(np.pi)

        # Compute inverse line width
        inverse_width = 1.0 / self.gaussian_width(temperature=temperature, v_turbulence=v_turbulence)
  
        # Get the index of the last spatial axis
        last_axis = abundance.dim() - 1

        # Compute the Doppler shift for each element
        shift = 1.0 + velocity_los * (1.0 / CC)

        # Create freqency tensor in the rest frame for each element
        freqs_restframe = torch.einsum("..., f -> ...f", shift, frequencies)

        # Define the a and b (tabulated) functions (Note: we absorb dx in the definition of a for efficiency)
        a = (1.0/sqrt_pi) * inverse_width * chi_ij * abundance
        b = torch.einsum("..., ...f -> ...f", inverse_width, freqs_restframe - self.frequency)

        # if a.isnan().any():
        #     raise Warning("NaNs in a.")
        # if b.isnan().any():
        #     raise Warning("NaNs in b.")

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

        # if a1.isnan().any():
        #     raise Warning("NaNs in a1.")
        # if a0.isnan().any():
        #     raise Warning("NaNs in a0.")
        # if b0.isnan().any():
        #     raise Warning("NaNs in b0.")
        # if b1.isnan().any():
        #     raise Warning("NaNs in b1.")
        # if sp_a1b0.isnan().any():
        #     raise Warning("NaNs in sp_a1b0.")
        # if sp_a0b1.isnan().any():
        #     raise Warning("NaNs in sp_a0b1.")

        dtau  = torch.einsum("..., ...f -> ...f", a1 - a0, expb0 - expb1)
        dtau += (sp_a1b0 - sp_a0b1) * (erfb0 - erfb1)
        dtau *= (0.5 / (b10**2 + 1.0e-30))
        # note that the regularizer 1.0e-30 is never going to be significant
        # however it is essential to avoid nans in backprop (see https://github.com/Magritte-code/pomme/issues/2)

        # if dtau.isnan().any():
        #     raise Warning("NaNs in dtau before mask.")

        # Create a mask for the region where the above calculation might have numerical issues
        mask_threshold = 1.0e-4
        mask           = torch.Tensor(torch.abs(b10) < mask_threshold)

        # Use a (second order) Taylor expansion in b10 for the masked region
        dtau[mask] = (   torch.einsum("..., ...f", (1.0/2.0) *  a01      , expb0           ) \
                       - torch.einsum("..., ...f", (1.0/3.0) * (a01 + a1), expb0 * b0 * b10) )[mask]

        # if dtau.isnan().any():
        #     raise Warning("NaNs in dtau after mask.")

        tau = torch.empty_like(b)
        tau[...,  0 , :] = 0.0
        tau[..., +1:, :] = torch.cumsum(dtau, dim=last_axis)

        return dtau, tau

    
    def image_along_last_axis(self, pop, abundance, temperature, v_turbulence, velocity_los, frequencies, dx, img_bdy):
        """
        Create an image along the last data axis.

        Parameters
        ----------
        pop : torch.Tensor
            Level populations for the line producing species.
        abundance : torch.Tensor
            Abundance distribution of the line producing species.
        temperature : torch.Tensor
            Temperature distribution of the line producing species.
        v_turbulence : torch.Tensor
            Turbulent velocity distribution.
        velocity_los : torch.Tensor
            Line of sight velocity distribution.
        frequencies : torch.Tensor
            Frequencies at which to image the model.
        dx : torch.Tensor
            Grid spacing along the line of sight.
        img_bdy : torch.Tensor
            Boundary conditions for the image.
    
        Returns
        -------
        img : torch.Tensor
            Tensor containing the image of the model.
        """
        # Compute the line emissivity and opacity from the level populations
        eta_ij, chi_ij = self.emissivity_and_opacity_ij(pop=pop)

        # if eta_ij.isnan().any() or chi_ij.isnan().any():
        #     raise Warning("NaNs in emissivity or opacity.")

        dtau, tau = self.optical_depth_along_last_axis(
            chi_ij       = chi_ij,
            abundance    = abundance,
            temperature  = temperature,
            v_turbulence = v_turbulence,
            velocity_los = velocity_los,
            frequencies  = frequencies,
            dx           = dx
        )

        # if dtau.isnan().any() or tau.isnan().any():
        #     raise Warning("NaNs in dtau or tau.")

        src = eta_ij / chi_ij
        img = forward_image_along_last_axis(src=src, dtau=dtau, tau=tau, img_bdy=img_bdy)

        # if img.isnan().any():
        #     raise Warning("NaNs in image.")

        return img

    # @torch.compile
    def LTE_image_along_last_axis(self, abundance, temperature, v_turbulence, velocity_los, frequencies, dx, img_bdy):
        """
        Create an image along the last data axis, assuming LTE level populations.

        Parameters
        ----------
        abundance : torch.Tensor
            Abundance distribution of the line producing species.
        temperature : torch.Tensor
            Temperature distribution of the line producing species.
        v_turbulence : torch.Tensor
            Turbulent velocity distribution.
        velocity_los : torch.Tensor
            Line of sight velocity distribution.
        frequencies : torch.Tensor
            Frequencies at which to image the model.
        dx : torch.Tensor
            Grid spacing along the line of sight.
        img_bdy : torch.Tensor
            Boundary conditions for the image.
        
        Returns
        -------
        img : torch.Tensor
            Tensor containing the image of the model.
        """

        pop = self.LTE_pops(
            temperature = temperature
        )

        # print_var('pop         ', pop)
        # print_var('abundance   ', abundance)
        # print_var('temperature ', temperature)
        # print_var('v_turbulence', v_turbulence)
        # print_var('velocity_los', velocity_los)
        # print_var('frequencies ', frequencies)

        img = self.image_along_last_axis(
            pop          = pop,
            abundance    = abundance,
            temperature  = temperature,
            v_turbulence = v_turbulence,
            velocity_los = velocity_los,
            frequencies  = frequencies,
            dx           = dx,
            img_bdy      = img_bdy
        )

        # print_var('img', img)

        return img


    def freq_to_velo(self, freq, unit='m/s'):
        """
        Convert frequencies with respect to this line to velocities.

        Parameters
        ----------
        freq : torch.Tensor
            Frequency tensor to convert.
        unit : str
            Unit of the velocity. (Default 'm/s'.)

        Returns
        -------
        out : torch.Tensor
            Tensor containing the velocities corresponding to the given frequencies.
        """
        return (freq / (self.frequency * units.Hz) - 1.0) * constants.c.to(unit).value