import numpy as np

from astroquery.lamda import Lamda, parse_lamda_datafile
from astropy          import constants


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