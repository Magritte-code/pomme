import torch
import numpy as np

from time             import time
from astroquery.lamda import Lamda, parse_lamda_datafile
from astropy          import constants
from p3droslo.utils   import get_molar_mass


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
        
        print(f"You have selected line:")
        print(f"    {self.species_name}(J={self.J_upper}-{self.J_lower})")
        print(f"Please check the properties that were inferred:")
        print(f"    {'Frequency '       :<17} {self.frequency :0.9e}  Hz")
        print(f"    {'Einstein A coeff ':<17} {self.Einstein_A:0.9e}  1/s")
        print(f"    {'Molar mass'       :<17} {self.species_molar_mass:<15}  g/mol")
        
        
    def gaussian_width(self, temperature, v_turbulence):
        """
        Gaussian spectral line width.
        """
        # Compute convenience variables
        factor_1 = self.frequency / CC
        factor_2 = 2.0 * KB / (self.species_molar_mass * AMU)
        # Return the gaussian line width
        return factor_1 * torch.sqrt(factor_2*temperature + v_turbulence**2)


    def gaussian_profile(self, temperature, v_turbulence, freq):
        """
        Gaussian spectral line profile function.
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
        LTE level populations for the given temperature.
    
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
    

    def opacity_ij(self, pop):
        """
        Line opacity, not folded with the profile.
        
        Parameters
        ----------
        temperature : torch.Tensor
            Temperature for which to evaluate the LTE level populations.
    
        Returns
        -------
        eta, chi : torch.Tensor, torch.Tensor
            Tensor containing the LTE emissivities and opacities for the given temperature.
        """
        # Compute the prefactor
        factor = HH * self.frequency / (4.0 * np.pi)
        
        # Compute the opacity
        chi = factor * (self.Einstein_Ba * pop[self.lower] - self.Einstein_Bs * pop[self.upper]) 
        
        # Return results
        return chi
    

    def LTE_emissivity_and_opacity(self, density, temperature, v_turbulence, frequencies):
        """
        Line emissivity and opacity assuming LTE.
        
        Parameters
        ----------
        temperature : torch.Tensor
            Temperature for which to evaluate the LTE level populations.
    
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
        # Fold the emessivities and opacities with the profile and the number density
        eta = torch.einsum("..., ...f -> ...f", eta*density, profile)
        chi = torch.einsum("..., ...f -> ...f", chi*density, profile)
        # t += time()
        # print("multiply ", t)
        
        # Return results
        return (eta, chi)